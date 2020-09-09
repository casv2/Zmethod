module MD

kB = 8.617330337217213e-05 #units
fs = 0.09822694788464063

using Distributions, LinearAlgebra
using ASE
using PyCall

@pyimport ase
ase_write = pyimport("ase.io")["write"]

export Stationary, VelocityVerlet, Zmethod, MaxwellBoltzmann_scale

function MaxwellBoltzmann(at::Atoms, T::Int)
    d = Normal()
    M = reshape(collect(rand(d, 3*length(at.M))), (length(at.M),3))
    M2 = M .* sqrt.(at.M .* T)
    set_momenta!(at, transpose(M2))

    return at
end

function Stationary(at::Atoms)
    p0 = sum(at.P)
    mtot = sum(at.M)
    v0 = p0 / mtot

    momenta = [at.M[i] * v0 for i in 1:length(at.M)]
    set_momenta!(at, at.P - momenta)

    return at
end

function VelocityVerlet(IP::AbstractCalculator, at::Atoms, dt::Float64)
    V = at.P ./ at.M
    A = forces(IP, at) ./ at.M

    set_positions!(at, at.X + (V .* dt) + (.5 * A * dt^2))

    nA = forces(IP, at) ./ at.M
    nV = V + (.5 * (A + nA) * dt)
    set_momenta!(at, nV .* at.M)

    return at
end

function Zmethod(IP::AbstractCalculator, at::Atoms, nsteps::Int, dt::Float64, A::Float64, N::Int, file::String)
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
                write_xyz("traj_$(i).xyz", pyat)
                push!(pyat_l, pyat)
            end
        end
    end

    return E_tot, E_pot, E_kin, P, T, pyat_l
end

function MaxwellBoltzmann_scale(at::Atoms, T::Float)
    d = Normal()
    M = reshape(collect(rand(d, 3*length(at.M))), (length(at.M),3))

    N = []

    for j in 1:1000
        d = Normal()
        M = reshape(collect(rand(d, 3*length(at.M))), (length(at.M),3))
        n = 0
        for i in 1:1:length(M[:,1])
            n += norm(M[i,:])
        end
        push!(N,n)
    end

    d = Normal()
    M = reshape(collect(rand(d, 3*length(at.M))), (length(at.M),3))
    Mnm = mean(N)

    for i in 1:100000
        n = 0
        for i in 1:length(M[:,1])
            n += norm(M[i,:])
        end
        if n < Mnm
            M = M*1.0001
        else
            M = M*0.9999
        end
    end

    M2 = M .* sqrt.(at.M .* T)
    set_momenta!(at, transpose(M2))

    return at #, Ml, Mnm, Sl, M
end

end
