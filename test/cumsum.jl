@loops function my_cumsum!(_, a, b, ptop, fac)
    let (irange, jrange) = (axes(a, 1), axes(a, 2))
        nz = size(a,3)
        for j in jrange
            @vec for i in irange
                @inbounds a[i, j, nz] = ptop + (fac/2)*b[i, j, nz]
            end
        end
        for j in jrange
            for k = nz:-1:2
                @vec for i in irange
                    @inbounds a[i, j, k-1] = a[i, j, k] + (b[i, j, k-1] + b[i,j,k])*(fac/2)
                end
            end
        end
    end
end

@loops function my_cumsum2!(_, a, b, ptop, fac)
    let (irange, jrange) = (axes(a, 1), axes(a, 2))
        nz = size(a,3)
        for j in jrange
            @vec for i in irange
                @inbounds a[i, j, nz] = ptop + (fac/2)*b[i, j, nz]
            end
            for k = nz:-1:2
                @vec for i in irange
                    @inbounds a[i, j, k-1] = a[i, j, k] + (b[i, j, k-1] + b[i,j,k])*(fac/2)
                end
            end
        end
    end
end
