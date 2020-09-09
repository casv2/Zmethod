module Zm

using Distributed, ASE, IPFitting
using Zmethod.MD: VelocityVerlet
using LinearAlgebra

export Zmethod

# function do_Zm(label::String, at::Atoms, V::Array{Float64}, nsteps::Int, A::Float64; dt=1, N=6)
#     @eval @everywhere label, at_file, V, nsteps, A, dt, N = $label, $at_file, $V, $nsteps, $A, $dt, $N
#
#     @sync @distributed for v in V
#         set_cell!(at, at.cell * v)
#         set_positions!(at, at.X * v)
#         E_tot, E_pot, E_kin, P, T, pyat_l = Zmethod.Zm.Zmethod(IP, at, nsteps, dt, A, N, label, write=false)
#         #write_xyz(label *"_$(v).xyz", pyat_l)
#     end
# end

kB = 8.617330337217213e-05 #units
fs = 0.09822694788464063

function Zmethod(IP::AbstractCalculator, at::Atoms, nsteps::Int, dt::Int, A::Float64, N::Int, file::String; write=false)
    E0 = energy(IP, at)

    m = at.M

    E_tot = zeros(nsteps)
    E_pot = zeros(nsteps)
    E_kin = zeros(nsteps)
    P = zeros(nsteps)
    T = zeros(nsteps)

    pyat_l = []

    open(file, "w") do io
        for i in 1:nsteps
            at = VelocityVerlet(IP, at, dt * fs)
            Ek = ((0.5 * sum(at.M) * norm(at.P ./ at.M)^2)/length(at.M)) / length(at.M)
            Ep = (energy(IP, at) - E0) / length(at.M)
            E_tot[i] = Ek + Ep
            E_pot[i] = Ep
            E_kin[i] = Ek
            T[i] = Ek / (1.5 * kB)
            P[i] = -tr(stress(IP, at))/3.0
            if i % 10 == 0
                write(io, "$(Ep) $(Ek) $(T[i]) $(P[i])\n")
            end

            v = at.P ./ m
            C = A/norm(v)

            set_momenta!(at, (v + C*v) .* m)

            if i % 100 == 0
                @show i, T[i], P[i]
                pyat = ASE.ASEAtoms(at)
                if write
                    write_xyz("traj_$(i).xyz", pyat)
                end
                push!(pyat_l, pyat)
            end
        end
    end

    return E_tot, E_pot, E_kin, P, T, pyat_l
end

end
