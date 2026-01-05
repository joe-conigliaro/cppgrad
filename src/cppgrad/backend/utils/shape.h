inline void coords_from_linear_fixed(size_t lin, const backend::View& v, uint32_t* coords) {
    // row-major unravel with v.shape[]
    for (int i = (int)v.rank - 1; i >= 0; --i) {
        const uint32_t dim = v.shape[i];
        coords[i] = dim ? (uint32_t)(lin % dim) : 0;
        lin = dim ? (lin / dim) : 0;
    }
}

inline size_t index_from_coords_fixed(const backend::View& v, const uint32_t* coords) {
    size_t idx = (size_t)v.offset;
    for (uint32_t i = 0; i < v.rank; ++i) idx += (size_t)coords[i] * (size_t)v.strides[i];
    return idx;
}

// broadcast mapping: right-align shapes like your AccessMeta::broadcast_from
inline void map_out_to_in_coords_broadcast_fixed(const uint32_t* outc,
                                                 const backend::View& vo,
                                                 const backend::View& vi,
                                                 uint32_t* inc) {
    // vi.rank may be <= vo.rank; align from the right
    for (uint32_t i = 0; i < vi.rank; ++i) inc[i] = 0;

    for (uint32_t oi = 0; oi < vo.rank; ++oi) {
        // map output axis oi -> input axis ii (right aligned)
        const int ii = (int)oi - ((int)vo.rank - (int)vi.rank);
        if (ii < 0) continue;

        // if input dim is 1, broadcast => coord 0; else copy coord
        inc[ii] = (vi.shape[ii] == 1) ? 0 : outc[oi];
    }
}
