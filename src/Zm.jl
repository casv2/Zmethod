module Zm


function do_Zm(at::Atoms, )

    function Zmethod(IP::AbstractCalculator, at::Atoms, nsteps::Int, dt::Float64, A::Float64, N::Int, file::String; write=false)
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
                    if write
                        pyat = ASE.ASEAtoms(at)
                        write_xyz("traj_$(i).xyz", pyat)
                    end
                    push!(pyat_l, pyat)
                end
            end
        end

        return E_tot, E_pot, E_kin, P, T, pyat_l
    end

end
