//
// Multi-column mul_mat_vec for reordered K-quant weights on the XMX engines (int8 DPAS).
// A thread owns 16 weight rows (B operand, one dword is 4 weights of a row), the q8_1 activation
// columns are the A operand. A super-block is 8 groups of 32, run as 2 parts of 4 groups. A quant
// type supplies its loads and unpack as a traits struct.
//
using sycl::ext::intel::esimd::simd;
constexpr int XMX_ROWS        = 16;  // weight rows per thread (DPAS N)
constexpr int XMX_CHUNK       = 64;  // widest single launch, more columns run as repeated launches
constexpr int XMX_FUSED_CHUNK = GGML_SYCL_XMX_GLU_MAX_COLS;  // widest launch of the fused gate and up kernel

// threads that split K for one tile, sized so that the shared memory stays small
template <int NC> constexpr int xmx_n_split() {
    return NC <= 8 ? 32 : (NC == 16 ? 16 : 8);
}

inline bool xmx_supported(const void * vx, const int ncols) {
    // 2D block loads need a 64 byte aligned base
    return ncols % QK_K == 0 && (uintptr_t) vx % 64 == 0;
}

// transposed 2D block load of 16 rows, dword (kd, n) lands at [kd * 16 + n]
// 2D block loads need rows of at least 64 bytes, shorter rows (SR) are read row by row
template <int W, bool SR>
ESIMD_INLINE simd<uint32_t, W * XMX_ROWS> xmx_load_t(const void * base, unsigned pitch, unsigned h, int x, int y) {
    using namespace sycl::ext::intel::esimd;
    if constexpr (SR) {
        simd<uint32_t, W * XMX_ROWS> r;
        for (int n = 0; n < XMX_ROWS; ++n) {
            const unsigned row                = y + n < (int) h ? y + n : h;
            const char *   p                  = (const char *) base + (size_t) row * (pitch + 1);
            r.template select<W, XMX_ROWS>(n) = block_load<uint32_t, W>((const uint32_t *) p + x);
        }
        return r;
    } else {
        return load_2d<uint32_t, W, XMX_ROWS, 1, true, false>((const uint32_t *) base, pitch, h, pitch, x, y);
    }
}

// per byte (q - zero) as int8 for q in 0..127, no borrow between bytes
ESIMD_INLINE simd<uint32_t, 128> xmx_bytes_sub(simd<uint32_t, 128> x, uint32_t zero_x4) {
    return ((x | 0x80808080u) - zero_x4) ^ 0x80808080u;
}

// How the scales of a quant type enter the DPAS epilogue of a group of 32 (see xmx_group)
enum class xmx_scale_kind {
    split,  // two scales per group, one per 16 wide half: the weights are masked to one half for each DPAS
    min,    // one scale and one min per group: a second DPAS sums the activations for the min term
    min16,  // a scale and a min per 16 wide half: the scales are folded into the DPAS operands
    plain,  // one scale per group
};

// What every quant type states about itself. nc_wide is the tile width from which it takes the large register file.
template <bool SR, int NC_WIDE, xmx_scale_kind KIND> struct xmx_traits_base {
    static constexpr bool short_rows = SR;
    static constexpr int  nc_wide    = NC_WIDE;
    static constexpr bool has_min    = KIND == xmx_scale_kind::min;
    static constexpr bool plain      = KIND == xmx_scale_kind::plain;
    static constexpr bool min16      = KIND == xmx_scale_kind::min16;
};

// The part of a context every quant type shares: the last row (the clamp of 2D loads), this thread's first row and
// the number of super-blocks per row. A type adds its array pointers and 2D load pitches.
struct xmx_ctx_base {
    unsigned h;
    int      row0;
    int      bpr;
};

ESIMD_INLINE void xmx_ctx_init(xmx_ctx_base & c, int nrows, int bpr, int row0) {
    c.h    = nrows - 1;
    c.row0 = row0;
    c.bpr  = bpr;
}

// pitch of a 2D load over an array with `bytes` bytes per super-block
constexpr unsigned xmx_pitch(int bpr, int bytes) {
    return bpr * bytes - 1;
}

// per group row scales of the 16 rows, one vector per group
template <int N> struct xmx_row_scales {
    simd<float, 16> v[N];

    ESIMD_INLINE simd<float, 16> get(int i) { return v[i]; }
};

// the 12 packed scale bytes of a super-block (Q3_K, Q4_K, Q5_K) of the 16 rows as three dwords
ESIMD_INLINE void xmx_load_sc12(const uint8_t *            sc,
                                const simd<uint32_t, 16> & bi,
                                simd<uint32_t, 16> &       w0,
                                simd<uint32_t, 16> &       w1,
                                simd<uint32_t, 16> &       w2) {
    using namespace sycl::ext::intel::esimd;
    w0 = gather<uint32_t, 16>((const uint32_t *) sc, bi * 12);
    w1 = gather<uint32_t, 16>((const uint32_t *) sc, bi * 12 + 4);
    w2 = gather<uint32_t, 16>((const uint32_t *) sc, bi * 12 + 8);
}

// d and dmin (a half2 per super-block, Q2_K, Q4_K, Q5_K) of the 16 rows
ESIMD_INLINE void xmx_load_dm(const sycl::half *          dm,
                              const simd<uint32_t, 16> & bi,
                              simd<float, 16> &          d,
                              simd<float, 16> &          dmin) {
    using namespace sycl::ext::intel::esimd;
    d    = convert<float>(gather<sycl::half, 16>(dm, bi * (2 * (uint32_t) sizeof(sycl::half))));
    dmin = convert<float>(
        gather<sycl::half, 16>(dm, bi * (2 * (uint32_t) sizeof(sycl::half)) + (uint32_t) sizeof(sycl::half)));
}

// Q6_K reorder layout: [ql: nb*128] [qh: nb*64] [scales int8: nb*16] [d: nb*2]
template <bool SR> struct xmx_traits_q6_k : xmx_traits_base<SR, 7, xmx_scale_kind::split> {
    struct ctx : xmx_ctx_base {
        const uint8_t *    ql;
        const uint8_t *    qh;
        const int8_t *     scales;
        const sycl::half * dh;
        unsigned           pl, ph, ps;
    };

    static ESIMD_INLINE ctx make(const void * vx, int nrows, int bpr, int row0) {
        const size_t nb = (size_t) nrows * bpr;
        ctx          c;
        xmx_ctx_init(c, nrows, bpr, row0);
        c.ql     = (const uint8_t *) vx;
        c.qh     = c.ql + nb * (QK_K / 2);
        c.scales = (const int8_t *) (c.qh + nb * (QK_K / 4));
        c.dh     = (const sycl::half *) (c.scales + nb * (QK_K / 16));
        c.pl     = xmx_pitch(bpr, QK_K / 2);
        c.ph     = xmx_pitch(bpr, QK_K / 4);
        c.ps     = xmx_pitch(bpr, QK_K / 16);
        return c;
    }

    // row scales of the 16 scale groups of a super-block
    struct scales_t {
        simd<int8_t, 256> sc8;
        simd<float, 16>   dv;

        ESIMD_INLINE simd<float, 16> get(int gr) {
            simd<int8_t, 16> s = sc8.template select<16, 4>(64 * (gr / 4) + (gr % 4));
            return sycl::ext::intel::esimd::convert<float>(s) * dv;
        }
    };

    // rows: clamped row index of each of the 16 lanes
    static ESIMD_INLINE scales_t load_scales(const ctx & c, int sb, const simd<uint32_t, 16> & rows) {
        using namespace sycl::ext::intel::esimd;
        scales_t s;
        s.dv = convert<float>(gather<sycl::half, 16>(c.dh, (rows * c.bpr + sb) * (uint32_t) sizeof(sycl::half)));
        simd<uint32_t, 64> raw = xmx_load_t<4, SR>(c.scales, c.ps, c.h, sb * 4, c.row0);
        s.sc8                  = raw.template bit_cast_view<int8_t>();
        return s;
    }

    struct part_t {
        simd<uint32_t, 128> a, b, h;
    };

    static ESIMD_INLINE part_t load_part(const ctx & c, int sb, int part) {
        part_t p;
        p.a = xmx_load_t<8, SR>(c.ql, c.pl, c.h, sb * 32 + part * 16, c.row0);
        p.b = xmx_load_t<8, SR>(c.ql, c.pl, c.h, sb * 32 + part * 16 + 8, c.row0);
        p.h = xmx_load_t<8, SR>(c.qh, c.ph, c.h, sb * 16 + part * 8, c.row0);
        return p;
    }

    static ESIMD_INLINE simd<uint32_t, 128> unpack(const part_t & p, int g, int /*part*/) {
        simd<uint32_t, 128> qs = (g & 1) ? p.b : p.a;
        simd<uint32_t, 128> x;
        if (g == 0) {
            x = (qs & 0x0F0F0F0Fu) | ((p.h << 4) & 0x30303030u);
        } else if (g == 1) {
            x = (qs & 0x0F0F0F0Fu) | ((p.h << 2) & 0x30303030u);
        } else if (g == 2) {
            x = ((qs >> 4) & 0x0F0F0F0Fu) | (p.h & 0x30303030u);
        } else {
            x = ((qs >> 4) & 0x0F0F0F0Fu) | ((p.h >> 2) & 0x30303030u);
        }
        return xmx_bytes_sub(x, 0x20202020u);
    }
};

// Q3_K reorder layout: [qs: nb*64] [hmask: nb*32] [scales: nb*12] [d: nb*2]
// chunk s of 32: 2 bits from qs byte 32*(s/4)+l at shift 2*(s%4), high bit is hmask[l] bit s, value is q - 4
template <bool SR> struct xmx_traits_q3_k : xmx_traits_base<SR, 16, xmx_scale_kind::split> {
    struct ctx : xmx_ctx_base {
        const uint8_t *    qs;
        const uint8_t *    hm;
        const uint8_t *    sc;
        const sycl::half * dh;
        unsigned           pq, ph;
    };

    static ESIMD_INLINE ctx make(const void * vx, int nrows, int bpr, int row0) {
        const size_t nb = (size_t) nrows * bpr;
        ctx          c;
        xmx_ctx_init(c, nrows, bpr, row0);
        c.qs = (const uint8_t *) vx;
        c.hm = c.qs + nb * (QK_K / 4);
        c.sc = c.hm + nb * (QK_K / 8);
        c.dh = (const sycl::half *) (c.sc + nb * 12);
        c.pq = xmx_pitch(bpr, QK_K / 4);
        c.ph = xmx_pitch(bpr, QK_K / 8);
        return c;
    }

    using scales_t = xmx_row_scales<16>;

    static ESIMD_INLINE scales_t load_scales(const ctx & c, int sb, const simd<uint32_t, 16> & rows) {
        using namespace sycl::ext::intel::esimd;
        scales_t           s;
        simd<uint32_t, 16> bi = rows * c.bpr + sb;
        simd<uint32_t, 16> w0, w1, w2;
        xmx_load_sc12(c.sc, bi, w0, w1, w2);
        simd<float, 16> dv = convert<float>(gather<sycl::half, 16>(c.dh, bi * (uint32_t) sizeof(sycl::half)));
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const simd<uint32_t, 16> lo   = (i % 8) < 4 ? w0 : w1;
            const int                bl   = i % 4;
            simd<uint32_t, 16>       nib  = (lo >> (8 * bl + (i < 8 ? 0 : 4))) & 0xFu;
            simd<uint32_t, 16>       hi   = ((w2 >> (8 * bl + 2 * (i / 4))) & 3u) << 4;
            simd<int, 16>            code = nib | hi;
            s.v[i]                        = convert<float>(code - 32) * dv;
        }
        return s;
    }

    struct part_t {
        simd<uint32_t, 128> q, h;
    };

    static ESIMD_INLINE part_t load_part(const ctx & c, int sb, int part) {
        part_t p;
        p.q = xmx_load_t<8, SR>(c.qs, c.pq, c.h, sb * 16 + part * 8, c.row0);
        p.h = xmx_load_t<8, SR>(c.hm, c.ph, c.h, sb * 8, c.row0);
        return p;
    }

    static ESIMD_INLINE simd<uint32_t, 128> unpack(const part_t & p, int g, int part) {
        const int           s = 4 * part + g;
        simd<uint32_t, 128> x = ((p.q >> (2 * g)) & 0x03030303u) | (((p.h >> s) & 0x01010101u) << 2);
        return xmx_bytes_sub(x, 0x04040404u);
    }
};

// Q4_K and Q5_K (QH) reorder layout: [qs: nb*128] ([qh: nb*32]) [scales: nb*12] [dm: nb*4]
// group j of 32: nibble of qs byte 32*(j/2)+l (low for even j), Q5_K adds 16 * bit j of qh[l]
// value is dall * sc_j * q - dmin * m_j, one scale and one min per group
template <bool QH, bool SR> struct xmx_traits_q45_k : xmx_traits_base<SR, QH ? 16 : 7, xmx_scale_kind::min> {
    struct ctx : xmx_ctx_base {
        const uint8_t *    qs;
        const uint8_t *    qh;
        const uint8_t *    sc;
        const sycl::half * dm;
        unsigned           pq, ph;
    };

    static ESIMD_INLINE ctx make(const void * vx, int nrows, int bpr, int row0) {
        const size_t nb = (size_t) nrows * bpr;
        ctx          c;
        xmx_ctx_init(c, nrows, bpr, row0);
        c.qs = (const uint8_t *) vx;
        c.qh = c.qs + nb * (QK_K / 2);
        c.sc = QH ? c.qh + nb * (QK_K / 8) : c.qh;
        c.dm = (const sycl::half *) (c.sc + nb * 12);
        c.pq = xmx_pitch(bpr, QK_K / 2);
        c.ph = xmx_pitch(bpr, QK_K / 8);
        return c;
    }

    struct scales_t {
        xmx_row_scales<8> d;  // dall * scale of each group
        xmx_row_scales<8> m;  // dmin * min of each group
    };

    static ESIMD_INLINE scales_t load_scales(const ctx & c, int sb, const simd<uint32_t, 16> & rows) {
        using namespace sycl::ext::intel::esimd;
        scales_t           s;
        simd<uint32_t, 16> bi = rows * c.bpr + sb;
        simd<uint32_t, 16> w0, w1, w2;
        xmx_load_sc12(c.sc, bi, w0, w1, w2);
        simd<float, 16> d, dmin;
        xmx_load_dm(c.dm, bi, d, dmin);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int          jj = j % 4;
            simd<uint32_t, 16> sc, m;
            if (j < 4) {
                sc = (w0 >> (8 * jj)) & 63u;
                m  = (w1 >> (8 * jj)) & 63u;
            } else {
                sc = ((w2 >> (8 * jj)) & 0xFu) | (((w0 >> (8 * jj + 6)) & 3u) << 4);
                m  = ((w2 >> (8 * jj + 4)) & 0xFu) | (((w1 >> (8 * jj + 6)) & 3u) << 4);
            }
            s.d.v[j] = convert<float>(simd<int, 16>(sc)) * d;
            s.m.v[j] = convert<float>(simd<int, 16>(m)) * dmin;
        }
        return s;
    }

    struct part_t {
        simd<uint32_t, 128> a, b, h;
    };

    static ESIMD_INLINE part_t load_part(const ctx & c, int sb, int part) {
        part_t p;
        p.a = xmx_load_t<8, SR>(c.qs, c.pq, c.h, sb * 32 + part * 16, c.row0);
        p.b = xmx_load_t<8, SR>(c.qs, c.pq, c.h, sb * 32 + part * 16 + 8, c.row0);
        if constexpr (QH) {
            p.h = xmx_load_t<8, SR>(c.qh, c.ph, c.h, sb * 8, c.row0);
        }
        return p;
    }

    static ESIMD_INLINE simd<uint32_t, 128> unpack(const part_t & p, int g, int part) {
        simd<uint32_t, 128> qs = (g < 2) ? p.a : p.b;
        simd<uint32_t, 128> x  = (qs >> (4 * (g & 1))) & 0x0F0F0F0Fu;
        if constexpr (QH) {
            x |= ((p.h >> (4 * part + g)) & 0x01010101u) << 4;
        }
        return x;
    }
};

// Q2_K reorder layout: [qs: nb*64] [scales: nb*16] [dm: nb*4]
// chunk s of 32: 2 bits from qs byte 32*(s/4)+l at shift 2*(s%4), each 16 wide half has a 4 bit scale and min
// value is d * (code & 15) * q - dmin * (code >> 4)
template <bool SR> struct xmx_traits_q2_k : xmx_traits_base<SR, 16, xmx_scale_kind::min16> {
    struct ctx : xmx_ctx_base {
        const uint8_t *    qs;
        const uint8_t *    sc;
        const sycl::half * dm;
        unsigned           pq, ps;
    };

    static ESIMD_INLINE ctx make(const void * vx, int nrows, int bpr, int row0) {
        const size_t nb = (size_t) nrows * bpr;
        ctx          c;
        xmx_ctx_init(c, nrows, bpr, row0);
        c.qs = (const uint8_t *) vx;
        c.sc = c.qs + nb * (QK_K / 4);
        c.dm = (const sycl::half *) (c.sc + nb * (QK_K / 16));
        c.pq = xmx_pitch(bpr, QK_K / 4);
        c.ps = xmx_pitch(bpr, QK_K / 16);
        return c;
    }

    struct scales_t {
        simd<uint8_t, 256> code;  // 4 bit scale (low) and min (high) of each 16 wide half
        simd<float, 16>    d, dmin;

        ESIMD_INLINE simd<uint8_t, 16> byte(int gr) { return code.template select<16, 4>(64 * (gr / 4) + (gr % 4)); }
    };

    static ESIMD_INLINE scales_t load_scales(const ctx & c, int sb, const simd<uint32_t, 16> & rows) {
        using namespace sycl::ext::intel::esimd;
        scales_t           s;
        simd<uint32_t, 16> bi  = rows * c.bpr + sb;
        simd<uint32_t, 64> raw = xmx_load_t<4, SR>(c.sc, c.ps, c.h, sb * 4, c.row0);
        s.code                 = raw.template bit_cast_view<uint8_t>();
        xmx_load_dm(c.dm, bi, s.d, s.dmin);
        return s;
    }

    struct part_t {
        simd<uint32_t, 128> q;
    };

    static ESIMD_INLINE part_t load_part(const ctx & c, int sb, int part) {
        part_t p;
        p.q = xmx_load_t<8, SR>(c.qs, c.pq, c.h, sb * 16 + part * 8, c.row0);
        return p;
    }

    static ESIMD_INLINE simd<uint32_t, 128> unpack(const part_t & p, int g, int /*part*/) {
        return (p.q >> (2 * g)) & 0x03030303u;
    }
};

// Q8_0 reorder layout: [qs: nrows*K bytes] [d: one half per 32 weights]
// the int8 weights feed DPAS as they are, a group is one block with one scale
template <bool SR> struct xmx_traits_q8_0 : xmx_traits_base<SR, 16, xmx_scale_kind::plain> {
    struct ctx : xmx_ctx_base {
        const uint8_t *    qs;
        const sycl::half * dh;
        unsigned           pq;
    };

    static ESIMD_INLINE ctx make(const void * vx, int nrows, int bpr, int row0) {
        ctx c;
        xmx_ctx_init(c, nrows, bpr, row0);
        c.qs = (const uint8_t *) vx;
        c.dh = (const sycl::half *) (c.qs + (size_t) nrows * bpr * QK_K);
        c.pq = xmx_pitch(bpr, QK_K);
        return c;
    }

    struct scales_t {
        xmx_row_scales<8> d;  // scale of each of the 8 blocks of a super-block
    };

    static ESIMD_INLINE scales_t load_scales(const ctx & c, int sb, const simd<uint32_t, 16> & rows) {
        using namespace sycl::ext::intel::esimd;
        scales_t           s;
        simd<uint32_t, 16> blk = (rows * c.bpr + sb) * (QK_K / QK8_0);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s.d.v[j] = convert<float>(gather<sycl::half, 16>(c.dh, (blk + j) * (uint32_t) sizeof(sycl::half)));
        }
        return s;
    }

    struct part_t {
        simd<uint32_t, 128> q[4];
    };

    static ESIMD_INLINE part_t load_part(const ctx & c, int sb, int part) {
        part_t p;
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            p.q[g] = xmx_load_t<8, SR>(c.qs, c.pq, c.h, sb * 64 + 8 * (4 * part + g), c.row0);
        }
        return p;
    }

    static ESIMD_INLINE simd<uint32_t, 128> unpack(const part_t & p, int g, int /*part*/) { return p.q[g]; }
};

template <bool SR> using xmx_traits_q4_k = xmx_traits_q45_k<false, SR>;
template <bool SR> using xmx_traits_q5_k = xmx_traits_q45_k<true, SR>;

// registers per thread, the large file halves the resident threads but avoids spills in wide tiles, the fused
// kernel keeps two sets of weights and scales and always takes it
template <typename T, int NC, bool FUSED> constexpr int xmx_grf() {
    return FUSED || NC >= T::nc_wide ? 256 : 128;
}

#ifdef GGML_SYCL_MMVQ_HAS_XMX
// One DPAS of a block of RC columns: the int32 products of the weights B and the activations A, as float and scaled
// per weight row. Every quant type sums one or two of these per group of 32.
template <int RC, int NT>
ESIMD_INLINE simd<float, RC * NT> xmx_term(const simd<int8_t, 512> &    B,
                                           const simd<int8_t, RC * 32> & A,
                                           const simd<float, NT> &       s) {
    using namespace sycl::ext::intel::esimd;
    namespace xmx = sycl::ext::intel::esimd::xmx;

    simd<int, RC * NT> C = xmx::dpas<8, RC, int>(B, A);
    return convert<float>(C) * s.template replicate<RC>();
}

// one group of 32 of one weight set, accumulated into Cf for all the columns of the tile
template <typename T, int NC, typename A2T, typename D8T>
ESIMD_INLINE void xmx_group(simd<float, NC * XMX_ROWS> & Cf,
                            const typename T::part_t &   pd,
                            typename T::scales_t &       sc,
                            A2T &                        A2,
                            D8T &                        d8f,
                            simd<int8_t, 512> &          Bl,
                            simd<int8_t, 512> &          Bh,
                            const simd<int8_t, 512> &    Ol,
                            const int                    g,
                            const int                    part) {
    using namespace sycl::ext::intel::esimd;

    constexpr int  NT    = XMX_ROWS;
    constexpr int  RC    = NC < 8 ? NC : 8;
    constexpr int  NB    = NC / RC;
    constexpr bool split = !T::has_min && !T::plain && !T::min16;  // two scales per group of 32
    constexpr bool pair  = (RC * 32) % 64 == 0;                    // a block of RC rows fills whole registers

    const int gi = 4 * part + g;

    simd<uint32_t, 128> x = T::unpack(pd, g, part);
    simd<int8_t, 512>   B = x.template bit_cast_view<int8_t>();
    if constexpr (split) {
        Bl.template select<256, 1>(0)   = B.template select<256, 1>(0);
        Bh.template select<256, 1>(256) = B.template select<256, 1>(256);
    }

    simd<float, NT> rd;
    simd<float, NT> rm;
    simd<float, NT> rs0;
    simd<float, NT> rs1;

    // Q2_K: scale and min of each 16 wide half are 4 bit codes, the weight is d * scl * q - dmin * mc. scl * q
    // is at most 45, so a dword multiply keeps the bytes apart and the scaled weights go to DPAS as int8. The
    // min term sums a * mc over K, the codes mc take the place of the weights in a second DPAS.
    simd<int8_t, 512> Bs;
    simd<int8_t, 512> Bm;
    if constexpr (T::min16) {
        const simd<uint8_t, NT> c0  = sc.byte(2 * gi);
        const simd<uint8_t, NT> c1  = sc.byte(2 * gi + 1);
        simd<uint32_t, NT>      sl0 = convert<uint32_t>(simd<uint8_t, NT>(c0 & (uint8_t) 15));
        simd<uint32_t, NT>      sl1 = convert<uint32_t>(simd<uint8_t, NT>(c1 & (uint8_t) 15));
        simd<uint32_t, NT>      mc0 = convert<uint32_t>(simd<uint8_t, NT>(c0 >> (uint8_t) 4)) * 0x01010101u;
        simd<uint32_t, NT>      mc1 = convert<uint32_t>(simd<uint8_t, NT>(c1 >> (uint8_t) 4)) * 0x01010101u;
        simd<uint32_t, 128>     ws;
        simd<uint32_t, 128>     wm;
        ws.template select<64, 1>(0)  = x.template select<64, 1>(0) * sl0.template replicate<4>();
        ws.template select<64, 1>(64) = x.template select<64, 1>(64) * sl1.template replicate<4>();
        wm.template select<64, 1>(0)  = mc0.template replicate<4>();
        wm.template select<64, 1>(64) = mc1.template replicate<4>();
        Bs                            = ws.template bit_cast_view<int8_t>();
        Bm                            = wm.template bit_cast_view<int8_t>();
    }
    if constexpr (T::plain) {
        rd = sc.d.get(gi);
    } else if constexpr (T::has_min) {
        rd = sc.d.get(gi);
        rm = sc.m.get(gi);
    } else if constexpr (T::min16) {
        // scales are applied through Bs and Bm
    } else {
        rs0 = sc.get(2 * gi);
        rs1 = sc.get(2 * gi + 1);
    }

#pragma unroll
    for (int b = 0; b < NB; ++b) {
        simd<int8_t, RC * 32> A =
            A2[b].template select<RC * 8, 1>(pair ? (g % 2) * RC * 8 : 0).template bit_cast_view<int8_t>();
        simd<float, RC * NT> d8r =
            d8f[b].template select<RC, RC == 1 ? 1 : 8>(gi).template replicate_vs_w_hs<RC, 1, NT, 0>(0);
        auto Cb = Cf.template select<RC * NT, 1>(b * RC * NT);

        simd<float, RC * NT> z;
        if constexpr (T::plain) {
            z = xmx_term<RC, NT>(B, A, rd);
        } else if constexpr (T::has_min) {
            // the second DPAS (all ones against the activations) is the exact integer sum of the activations
            z = xmx_term<RC, NT>(B, A, rd) - xmx_term<RC, NT>(Ol, A, rm);
        } else if constexpr (T::min16) {
            z = xmx_term<RC, NT>(Bs, A, sc.d) - xmx_term<RC, NT>(Bm, A, sc.dmin);
        } else {
            z = xmx_term<RC, NT>(Bl, A, rs0) + xmx_term<RC, NT>(Bh, A, rs1);
        }
        Cb += z * d8r;
    }
}

// activation of the GLU applied to the gate, uniform across the launch
template <int N> ESIMD_INLINE simd<float, N> xmx_glu_act(const simd<float, N> & gate, const int glu_op) {
    using namespace sycl::ext::intel::esimd;
    if (glu_op == GGML_GLU_OP_SWIGLU) {
        return gate / (1.0f + exp(-gate));
    }
    // GEGLU, the tanh is written with exp
    simd<float, N> a = 0.79788456f * gate * (1.0f + 0.044715f * gate * gate);
    simd<float, N> t = 1.0f - 2.0f / (exp(2.0f * a) + 1.0f);
    return 0.5f * gate * (1.0f + t);
}

// activations come in through 2D block loads, one for the (d, s) pairs of a super-block (s is unused) and one per two
// groups for the quants, the DPAS A operand needs no repacking. Groups with two scales (16 wide) mask the
// weights instead, zeros in B drop the other half of K.
// KS is the most threads that split K for one tile, the launch may use fewer.
// FUSED computes up * act(gate) from the weights vx (up) and vg (gate), reading the activations once.
template <typename T, int NC, int KS, bool FUSED>
ESIMD_INLINE void xmx_mul_mat(const void * vx,
                              const void * vg,
                              const char * vy,
                              float *      dst,
                              const int    ncols,
                              const int    nrows,
                              const int    ncv,
                              const int    stride_y_bytes,
                              const int    ldd,
                              const int    glu_op,
                              const int    tile,
                              const int    lid,
                              const int    ks) {
    using namespace sycl::ext::intel::esimd;

    constexpr int  NT    = XMX_ROWS;
    constexpr int  RC    = NC < 8 ? NC : 8;
    constexpr int  NB    = NC / RC;
    constexpr int  N_ACC = FUSED ? 2 : 1;
    constexpr bool pair  = (RC * 32) % 64 == 0;  // a block of RC rows fills whole registers

    if constexpr (KS > 1) {
        slm_init<N_ACC * KS * NC * NT * sizeof(float)>();
    }

    const int  bpr  = ncols / QK_K;
    const int  row0 = tile * NT;
    const auto c    = T::make(vx, nrows, bpr, row0);
    const auto cg   = T::make(FUSED ? vg : vx, nrows, bpr, row0);

    simd<uint32_t, NT> rows;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        rows[n] = row0 + n < nrows ? row0 + n : nrows - 1;
    }

    const uint32_t * yq  = (const uint32_t *) vy;
    const uint32_t * yd  = (const uint32_t *) (vy + ncols);
    const unsigned   yqw = ncols - 1;
    const unsigned   ydw = ncols / QK8_1 * sizeof(sycl::half2) - 1;
    const unsigned   yh  = ncv - 1;
    const unsigned   yp  = stride_y_bytes - 1;

    simd<float, NC * NT> Cf  = 0.0f;
    simd<float, NC * NT> Cfg = 0.0f;  // gate, only used when fused
    simd<int8_t, 512>    Bl  = int8_t(0);
    simd<int8_t, 512>    Bh  = int8_t(0);
    simd<int8_t, 512>    Ol  = int8_t(0);
    if constexpr (T::has_min) {
        Ol = int8_t(1);
    }

    for (int sb = lid; sb < bpr; sb += ks) {
        auto                 sc = T::load_scales(c, sb, rows);
        typename T::scales_t scg;
        if constexpr (FUSED) {
            scg = T::load_scales(cg, sb, rows);
        }

        // d of the 8 groups of the super-block, [column * 8 + group]
        simd<float, 8 * RC> d8f[NB];
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            simd<uint32_t, 8 * RC>    ds = load_2d<uint32_t, 8, RC, 1, false, false>(yd, ydw, yh, yp, sb * 8, b * RC);
            simd<sycl::half, 16 * RC> dh = ds.template bit_cast_view<sycl::half>();
            d8f[b]                       = convert<float>(simd<sycl::half, 8 * RC>(dh.template select<8 * RC, 2>(0)));
        }

#pragma unroll
        for (int part = 0; part < 2; ++part) {
            const auto         pd = T::load_part(c, sb, part);
            typename T::part_t pdg;
            if constexpr (FUSED) {
                pdg = T::load_part(cg, sb, part);
            }

            simd<uint32_t, 16 * RC> A2[NB];  // quants of two groups

#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const int gi = 4 * part + g;
                if constexpr (pair) {
                    if (g % 2 == 0) {
#pragma unroll
                        for (int b = 0; b < NB; ++b) {
                            A2[b] =
                                load_2d<uint32_t, 8, RC, 2, false, false>(yq, yqw, yh, yp, sb * 64 + gi * 8, b * RC);
                        }
                    }
                } else {
#pragma unroll
                    for (int b = 0; b < NB; ++b) {
                        A2[b].template select<RC * 8, 1>(0) =
                            load_2d<uint32_t, 8, RC, 1, false, false>(yq, yqw, yh, yp, sb * 64 + gi * 8, b * RC);
                    }
                }

                xmx_group<T, NC>(Cf, pd, sc, A2, d8f, Bl, Bh, Ol, g, part);
                if constexpr (FUSED) {
                    xmx_group<T, NC>(Cfg, pdg, scg, A2, d8f, Bl, Bh, Ol, g, part);
                }
            }
        }
    }

    if constexpr (KS > 1) {
        slm_block_store<float, NC * NT>(lid * NC * NT * sizeof(float), Cf);
        if constexpr (FUSED) {
            slm_block_store<float, NC * NT>((KS + lid) * NC * NT * sizeof(float), Cfg);
        }
        barrier();
        if (lid != 0) {
            return;
        }
        Cf  = 0.0f;
        Cfg = 0.0f;
#pragma unroll 4
        for (int j = 0; j < ks; ++j) {
            Cf += slm_block_load<float, NC * NT>(j * NC * NT * sizeof(float));
            if constexpr (FUSED) {
                Cfg += slm_block_load<float, NC * NT>((KS + j) * NC * NT * sizeof(float));
            }
        }
    }

    if constexpr (FUSED) {
        Cf *= xmx_glu_act<NC * NT>(Cfg, glu_op);
    }

    if (row0 + NT <= nrows && ldd % 4 == 0 && (uintptr_t) dst % 16 == 0) {
#pragma unroll
        for (int m = 0; m < NC; ++m) {
            if (NC <= 8 || m < ncv) {
                block_store<float, NT>(dst + (size_t) m * ldd + row0, Cf.template select<NT, 1>(m * NT));
            }
        }
    } else {
        // tail tile or unaligned output
        simd<uint32_t, NT> lane(0, 1);
        simd_mask<NT>      in_rows = (lane + row0) < (uint32_t) nrows;
#pragma unroll
        for (int m = 0; m < NC; ++m) {
            if (NC <= 8 || m < ncv) {
                scatter<float, NT>(dst + (size_t) m * ldd + row0, lane * (uint32_t) sizeof(float),
                                   Cf.template select<NT, 1>(m * NT), in_rows);
            }
        }
    }
}

// KS_MAX bounds the threads that split K for one tile, the launch uses as few as keep every thread to the same
// number of super-blocks as the busiest one
template <typename T, int NC, int KS_MAX, bool FUSED>
static void xmx_mul_mat_launch_ks(const void *    vx,
                                  const void *    vg,
                                  const char *    vy,
                                  float *         dst,
                                  const int       ncols,
                                  const int       nrows,
                                  const int       ncv,
                                  const int       stride_y_bytes,
                                  const int       ldd,
                                  const int       glu_op,
                                  dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const size_t n_tiles = (nrows + XMX_ROWS - 1) / XMX_ROWS;
    const int    bpr     = ncols / QK_K;
    const int    rounds  = (bpr + KS_MAX - 1) / KS_MAX;
    const int    ks      = (bpr + rounds - 1) / rounds;
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(n_tiles * ks), sycl::range<1>(ks)),
                         sycl::ext::oneapi::experimental::properties{
                             sycl::ext::intel::experimental::grf_size<xmx_grf<T, NC, FUSED>()> },
                         [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                             xmx_mul_mat<T, NC, KS_MAX, FUSED>(vx, vg, vy, dst, ncols, nrows, ncv, stride_y_bytes, ldd,
                                                               glu_op, (int) it.get_group(0), (int) it.get_local_id(0),
                                                               ks);
                         });
    });
}

template <typename T, int NC, bool FUSED>
static void xmx_mul_mat_launch(const void *    vx,
                               const void *    vg,
                               const char *    vy,
                               float *         dst,
                               const int       ncols,
                               const int       nrows,
                               const int       ncv,
                               const int       stride_y_bytes,
                               const int       ldd,
                               const int       glu_op,
                               dpct::queue_ptr stream) {
    // the fused kernel keeps two sets of accumulators in shared memory, half the threads for the same size
    constexpr int ks_max = xmx_n_split<NC>() / (FUSED ? 2 : 1);
    xmx_mul_mat_launch_ks<T, NC, ks_max, FUSED>(vx, vg, vy, dst, ncols, nrows, ncv, stride_y_bytes, ldd, glu_op,
                                                stream);
}

// FUSED: vx and vg are the up and gate weights, dst gets up * act(gate) with glu_op a ggml_glu_op
template <typename T, bool FUSED = false>
static void xmx_mul_mat_ncols(const void *    vx,
                              const void *    vg,
                              const char *    vy,
                              float *         dst,
                              const int       ncols,
                              const int       nrows,
                              const int       ncols_dst,
                              const int       stride_y_bytes,
                              const int       ldd,
                              const int       glu_op,
                              dpct::queue_ptr stream) {
    constexpr int chunk_max = FUSED ? XMX_FUSED_CHUNK : XMX_CHUNK;
    for (int c0 = 0; c0 < ncols_dst; c0 += chunk_max) {
        const int    n  = std::min(chunk_max, ncols_dst - c0);
        const char * y  = vy + (size_t) c0 * stride_y_bytes;
        float *      d  = dst + (size_t) c0 * ldd;
        const int    nc = n > 32 ? 64 : (n > 16 ? 32 : (n > 8 ? 16 : n));
        switch (nc) {
#define XMX_CASE(N)                                                                                          \
    case N:                                                                                                  \
        xmx_mul_mat_launch<T, N, FUSED>(vx, vg, y, d, ncols, nrows, n, stride_y_bytes, ldd, glu_op, stream); \
        break
            XMX_CASE(1);
            XMX_CASE(2);
            XMX_CASE(3);
            XMX_CASE(4);
            XMX_CASE(5);
            XMX_CASE(6);
            XMX_CASE(7);
            XMX_CASE(8);
#undef XMX_CASE
            // short row variants are for tests, not built for the wide tiles
#define XMX_CASE_WIDE(N)                                                                                         \
    case N:                                                                                                      \
        if constexpr (!T::short_rows && (!FUSED || N <= XMX_FUSED_CHUNK)) {                                      \
            xmx_mul_mat_launch<T, N, FUSED>(vx, vg, y, d, ncols, nrows, n, stride_y_bytes, ldd, glu_op, stream); \
        }                                                                                                        \
        break
            XMX_CASE_WIDE(16);
            XMX_CASE_WIDE(32);
            XMX_CASE_WIDE(64);
#undef XMX_CASE_WIDE
            default:
                GGML_ABORT("unsupported ncols_dst=%d for XMX MMV", n);
        }
    }
}

#endif // GGML_SYCL_MMVQ_HAS_XMX
