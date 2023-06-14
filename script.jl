using Quantica
using LinearAlgebra, StaticArrays, SparseArrays


# useful functions
function fractional(x)
    return x - ceil(x-0.5)
end

function pdistance(r, A, Ainv)
    
    u = Ainv*r
    
    return A*fractional.(u)
end

# struct definition
struct LatticeBasis{E, D, Ts, ED}
    matrix::SMatrix{E, D, Ts, ED}
end

struct HoppingList{Tv, Ti}
    rowval::Vector{Ti}
    colval::Vector{Ti}
    nzval::Vector{Tv}
end

struct PeierlsHamiltonian{E, D, Ts, ED, Tv, Ti}
    bravais::LatticeBasis{E, D, Ts, ED}
    sites::Vector{SVector{E, Ts}}
    hoppings::HoppingList{Tv, Ti}
    separations::Vector{SVector{E, Ts}}
    matrix::SparseMatrixCSC{Tv, Ti}
end

function PeierlsHamiltonian(h::Quantica.Hamiltonian{T, E, D}) where {T, E, D}
    
    A = SMatrix{E, D}(h.lattice.bravais.matrix)
    Ainv = pinv(A)
    sites = h.lattice.unitcell.sites
    
    Is, Js, Vs = findnz(h[()])
    
    for n in 2:length(h.harmonics)
        
        In, Jn, Vn = findnz(h.harmonics[n].h.flat)
        
        append!(Is, In)
        append!(Js, Jn)
        append!(Vs, Vn)
    end
    
    hsp = sparse(Is, Js, Vs)
    
    rows, cols, vals = findnz(hsp)
    
    separations = [pdistance(sites[ij[1]] - sites[ij[2]], A, Ainv) for ij in zip(rows, cols)]
    
    return PeierlsHamiltonian(LatticeBasis(A), sites, HoppingList(rows, cols, vals), separations, hsp)
end

function (ph::PeierlsHamiltonian)(A)
    
    for i in eachindex(ph.matrix.nzval)
        
        ph.matrix.nzval[i] = cis(dot(ph.separations[i], A))*ph.hoppings.nzval[i]
        
    end
    
    return ph.matrix
end


# test: HBN
a1 = SA[0.5, sqrt(3)/2, 0]
a2 = SA[-0.5, sqrt(3)/2, 0]
sA = SA[0.0, 0.0, 0.0]
sB = (a1 + a2)/3;

lat = lattice(sublat(sA, name = :B), sublat(sB, name = :N), bravais = (a1, a2))
model = hopping(-1.0, range = 1.1/sqrt(3)) + onsite(0.1, sublats = :B) + onsite(-0.1, sublats = :N)

h = hamiltonian(lat, model) |> supercell(100, 100)
peierls! =  @hopping!((t, r, dr; A) -> t * cis(dot(A, dr)));
qh = hamiltonian(h, peierls!)
qh(A = SA[1, 2, 0])

ph = PeierlsHamiltonian(h);
ph(SA[1, 2, 0])

