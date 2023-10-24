### A Pluto.jl notebook ###
# v0.19.20

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ b2cf3f80-9e12-11ed-2cd7-fddcd14709e3
begin
	using POMDPs
	using POMDPTools
	using Distributions
	using Parameters
	using Random
end

# ╔═╡ 831a51e7-cd0e-4216-87df-263cdf7acc24
begin
	using ImageCore
	using ImageShow
	using Images
	using Compose
	using Cairo
	using ImageMagick
	using PlutoUI
	using Plots
	using StatsPlots
end

# ╔═╡ fdde7445-a477-46ea-a0c7-21e7f258858c
using FiniteDiff

# ╔═╡ 9ba3a0a3-433e-41d1-897a-c880b2da7569
using LinearAlgebra

# ╔═╡ e82b0afc-4072-4ae5-aca3-dd0542165469
md"""
# AA228: Policy Search
"""

# ╔═╡ a7fb576c-afc5-4be2-a5be-b722b312f6d0
md"""
So far, we've seen how exact solution methods can be used to solve for a policy in an offline method. We've also seen how online planning can handle large state spaces by reasoning over actions from an initial state. In this notebook, we'll discuss **policy search**, which involves searching over the space of policy parameters rather than actions. This idea will carry forward through the next few lectures.
"""

# ╔═╡ 18d9bf07-2a48-4bac-8a72-c46b5a0e7baf
TableOfContents()

# ╔═╡ 48b00590-f5c2-4ec1-95d3-7da1dd861e88
md"""
## Motivation
"""
# Consider an MDP with two states ($s_1$, $s_2$) and two actions ($a_1$, $a_2$). Draw the forward search tree up to depth three. What is the computational complexity of forward search?

# ╔═╡ 0cd75ebe-a86a-471b-8b13-2180f0bd25fa
md"""
**Q: Given the solution methods we've seen so far, how can we solve an MDP with continuous states or actions?**
"""

# ╔═╡ 1fce7b86-8413-499d-ae3a-f2a95496a0e2
# md"""
# A: 
# * Discrete MDP
# * Approximate value functions
# """

# ╔═╡ 24a02196-09b0-4249-8a5b-42b3217e8ad9
# md"""
# * One option is discretization, however we risk losing fidelity
# * In value function based methods, we can use approximate value iteration.
# * A common approach is to parameterize a policy. Then we can optimize over the policy parameters, which are usually much lower dimension than the action space.
# """

# ╔═╡ 76a5ba49-eff1-4aff-80eb-156963702404
md"""
Policy search helps us scale to very large or continuous state spaces.
"""

# ╔═╡ 49198ec3-6f5f-43a9-af14-eaae60e81142
md"""
## MDP Formulation: Inverted Pendulum
Let's define a running example with continuous states and actions to get started.

The inverted pendulum problem involves stabilizing a pendulum in an inverted position. Suppose we have a motor mounted at the pendulum's pivot point. The objective is to control the angle of the pendulum so that it stays as close to vertical as possible despite any disturbances that may occur, such as gusts of wind. This problem is challenging because the inverted pendulum is inherently unstable, making it difficult to maintain its balance.

Let's define the angle of the pendulum from vertical as $ϕ$.

We'll assume the pendulum has some fixed length $l$ and mass $m$.
"""

# ╔═╡ f930bfce-0810-4748-82a9-9004176be619
md"""
**Q: What should we use as the MDP state? Actions?**
"""
# * State: Angle and angular velocity of pendulum $s_t=(\phi_t, \dot{\phi}_t)$
# * Action: Motor torque ($a_t$)

# ╔═╡ 85ecd311-6760-4b65-ad98-6f0d69eab436
# md"""
# A:

# * State: Angle information $\phi$ and $\dot{\phi}$
# * Action: Motor torque
# """

# ╔═╡ 5b94e6b1-b6ef-4428-9f24-c061264a8cee
md"""
**Q: What is the transition model? (The actual equations are not important, but describe in words.)**
"""
# * The transition model is given by the dynamics (physics) of the pendulum. If you are curious the actual equations are:
# $\dot{\phi}_{t+1} = \dot{\phi}_t +  (-3g / (2l) \sin(\phi + π) + 3a / (ml^2))dt$
# $\phi_{t+1} = \phi_t + dt\dot{\phi}_{t+1}$

# ╔═╡ e35204bd-ca5e-45c0-94b5-0507575c9984
# md"""
# A: 
# * Physics based model
# """

# ╔═╡ 28f3b33c-1298-4d5b-8bbc-5a7af55e5b1e
md"""
**Q: What components might our reward function have?**
"""
# * We might want to penalize states based on the angle from vertical (maximum reward when the pendulum is vertical)
# * We would also want to penalize states with high velocity (maximum reward when the pendulum is stationary)
# * Finally, we could also penalize large actions. We might want our policy to minimize energy use.

# ╔═╡ 6ccd55cb-13f0-4ab6-a2f8-d12eae87e159
# md"""
# A:
# * Penalize angles far from vertical
# * Penalize based on angular velocity
# * Penalize large torques
# """

# ╔═╡ 4dc17f13-c1fa-43c5-a5aa-0c13f4062ed7
md"""
Great! Now let's look at how we can set define this MDP in code. We'll use the POMDPs.jl environment.

First, we'll define a struct type. This is a 
"""

# ╔═╡ 2d181244-cfe5-4158-880d-b920b14320db
@with_kw struct PendulumMDP <: MDP{Array{Float64}, Array{Float64}}
    Rstep = 1 # Reward earned on each step of the simulation
    λcost = 1 # Coefficient to the traditional OpenAIGym Reward
    max_speed::Float64 = 8.
    max_torque::Float64 = 100.
    dt::Float64 = .05
    g::Float64 = 10.
    m::Float64 = 1.
    l::Float64 = 1.
    γ::Float64 = 0.99
end

# ╔═╡ 299c6844-27d6-4488-b18d-1f0b8796b025
POMDPs.discount(mdp::PendulumMDP) = mdp.γ

# ╔═╡ 6b3385a3-418c-491d-9bff-bf4dc9c2ff5d
md"""
For many problems, explicitly writing the transition model $T(s' \mid s, a)$ and reward function $R(s,a)$ can be difficult. Here, we define a **generative model** of the dynamics and reward.
"""

# ╔═╡ f05e13d5-9e91-420c-8f38-72509d5a6723
angle_normalize(x) = mod((x+π), (2*π)) - π;

# ╔═╡ 59dc4f6a-0633-4a14-acef-4f6083ccd058
function pendulum_dynamics(env, s, a; rng::AbstractRNG = Random.GLOBAL_RNG)        
    θ, ω = s[1], s[2]
    dt, g, m, l = env.dt, env.g, env.m, env.l

    a = a[1]
    a = clamp(a, -env.max_torque, env.max_torque)
    costs = angle_normalize(θ)^2 + 0.1f0 * ω^2 + 0.001f0 * a^2

    ω = ω + (-3. * g / (2 * l) * sin(θ + π) + 3. * a / (m * l^2)) * dt
    θ = angle_normalize(θ + ω * dt)
    ω = clamp(ω, -env.max_speed, env.max_speed)

    sp = [θ, ω]
    r = env.Rstep - env.λcost*costs
    return sp, r
end;

# ╔═╡ 021b3e10-1d4c-4e08-8d09-5556103ebb46
function POMDPs.gen(mdp::PendulumMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG)
    sp, r = pendulum_dynamics(mdp, s, a, rng=rng)
    (sp = sp, r = r)
end

# ╔═╡ 9fb8392c-ae88-4829-8525-20d920e8c6b5
md"""
Define an initial state distribution
"""

# ╔═╡ 992f0385-8d35-49c9-ab8b-14bd8e33b4ef
function POMDPs.initialstate(mdp::PendulumMDP)
	θ0 = Distributions.Uniform(-π/6., π/6.)
	ω0 = Distributions.Uniform(-0.1, 0.1)
    ImplicitDistribution((rng) -> [rand(rng, θ0), rand(rng, ω0)])
end

# ╔═╡ a2784cb8-c534-4c41-8254-342f5eb16b9f
md"""
Rendering functions are below (hidden cells). Feel free to look under the hood if you are curious!
"""

# ╔═╡ 5342b6a4-f8ba-46be-baaa-be2ef3b4b9ec
function POMDPTools.render(mdp::PendulumMDP, step)
	s = step[:s]
	a = step[:a][1]
    θ = s[1] + π/2.
    point_array = [(0.5,0.5), (0.5 + 0.3*cos(θ), 0.5 - 0.3*sin(θ))]
    
    a_rad = abs(a)/10.
    if a < 0
        θstart = -3π/4
        θend = -θstart
        θarr = θend
    else
        θstart = π/4
        θend = -θstart
        θarr = θstart
    end 
    
    
    # Draw the arrow 
    endpt = (0.5, 0.5) .+ a_rad.*(cos(θarr), sin(θarr))
    uparr = endpt .+ 0.1*a_rad.*(cos(θarr)-sign(a)*sin(θarr), sign(a)*cos(θarr)+sin(θarr))
    dwnarr = endpt .+ 0.1*a_rad.*(-cos(θarr)-sign(a)sin(θarr), sign(a)*cos(θarr)-sin(θarr))
    arrow_array = [[endpt, uparr], [endpt, dwnarr]]
    
    
    img = compose(context(),
        (context(), Compose.line(arrow_array), Compose.arc(0.5, 0.5, a_rad, θstart, θend),  linewidth(0.5Compose.mm), fillopacity(0.), Compose.stroke("red")),
        (context(), Compose.circle(0.5, 0.5, 0.01), fill("blue"), Compose.stroke("black")),
        (context(), Compose.line(point_array), Compose.stroke("black"), linewidth(1Compose.mm)),
        (context(), Compose.rectangle(), fill("white"))
    )

	#return img
    tmpfilename = tempname()
    img |> PNG(tmpfilename, 10cm, 10cm)
    load(tmpfilename)
end

# ╔═╡ 2b045ae4-1176-44db-be0b-042f788b8e2c
function animate_pendulum(mdp, policy, fname)
	imgs = []
	s = rand(initialstate(mdp))
	for _=1:100
		a = action(policy, s)

		push!(imgs, render(mdp, (;s=s, a=a)))
		
		sp, r = @gen(:sp, :r)(mdp, s, a)
		s = sp
	end
	Images.save(fname, cat(imgs..., dims=3))
end

# ╔═╡ 74a1237d-0d70-4686-b0fd-d3b4e41be2d7
md"""
## Policy Parameterization
Now that we've established the components of our MDP, let's start to think about how to solve it. We already discussed how offline methods and online tree-search methods might be difficult to apply.


We introduce the notion of a **parameterized policy**. We can denote the action of policy $\pi$ at state $s$ parameterized by $\theta$ as

$a = \pi_{\theta}(s)$
for deterministic policies, and

$a \sim \pi_{\theta}(a \mid s)$
for stochastic policies.

**Policy space is often lower-dimensional than state space, and can often be searched more easily.**

The parameters θ may be a vector or some other more complex representation. For
example, we may want to represent our policy using a neural network with a
particular structure. We would use θ to represent the weights in the network.


"""

# ╔═╡ cc29ce34-7719-404a-81c8-22af6e44b680
md"""
**Q: How could we parameterize a policy for the inverted pendulum problem? Assume our state vector is $s=[\phi, \dot{\phi}]$**
"""

# ╔═╡ 5650c7cd-a977-4d4d-b63e-ba8db23bcefc
# md"""
# A: 

# $a = \theta_1 s_1  + \theta_2 s_2 = \theta^Ts$

# $a = \theta_1 s_1^{\theta_2}$
# """

# ╔═╡ 82804c10-2a66-41a8-9eec-96282588c386
# md"""
# **A:** One option is a weighted combination of elements in the state vector

# $\pi_{\theta}(s) = \theta_1 s_1 + \theta_2 s_2 = \theta^T s$

# This is a deterministic policy. If we wanted a stochastic policy, we could also use a linear gaussian model.

# $\pi_{\theta}(a \mid s) = \mathcal{N}(a \mid \theta_A s + \theta_b, \theta_{\Sigma})$
# """

# ╔═╡ aa1265ae-1b99-41ef-bce4-f24c946d066f
md"""
## Policy Evaluation
The expected discounted return of a policy $\pi$ from initial state distribution $b(s)$ is

$U(\pi) = \sum_s b(s) U^{\pi}(s)$

When we have a large or continuous state space, we often cannot compute the utility of following a policy $U(\pi)$ exactly. Instead, we rewrite $U(\pi)$ in terms of trajectories of states, actions, and rewards under the policy $\pi$.  If $R(\tau)$ is the sum of discounted rewards for trajectory $\tau$,

$U(\pi)= \int p_{\pi}(\tau)R(\tau)dt$

The expected value (or mean) total discounted reward can be _approximated_ by taking the mean total reward of many trajectories.

$U(\pi) \approx \frac{1}{m} \sum_{i=1}^m R(\tau^{(i)})$

This is sometimes called Monte Carlo policy evaluation.
"""

# ╔═╡ 00aedb7b-c0df-4344-acc8-2bb1cdc83db6
md"""
The POMDPs.jl package provides us with a convenient way to get the sum of discounted returns from a simulation (or a 'rollout'). The $\texttt{RolloutSimulator}$ type simulates a given policy for a fixed number of steps and returns the sum of discounted returns. Once we create a simulator like:
$\texttt{sim = RolloutSimulator(max\_steps=max\_steps)}$

We can compute get the total discounted reward using 
$\texttt{R=simulate(sim, mdp, policy)}$
"""

# ╔═╡ a4e51e09-6288-425a-82af-f6dd4e019d1b
md"""
**Let's write a function to perform Monte Carlo policy evaluation.**
"""

# ╔═╡ b0f8a206-ee75-4d6c-bd19-8ae840df46b2
function mc_policy_evaluation(mdp::MDP, π::Policy; m=100, max_steps=100)
	sim = RolloutSimulator(max_steps=max_steps)
	return mean([simulate(sim, mdp, π) for _=1:m])
end;

# ╔═╡ 33a1f2a2-188a-4674-b49d-0346f23449e8
md"""
## Policy Search Overview

In policy search, our goal is to optimize a policy's utility with respect to its parameters. In other words, we search over the parameter space for a set of parameters that maximize our utility.

Here, we'll try out policy search for a simple 2D policy parameterization:

$\pi_{\theta}(s) = \theta_1 s_1 + \theta_2 s_2 = \theta^T s$

First, let's create our MDP
"""

# ╔═╡ 50018cd7-8427-44fd-aeb2-dc25853d912c
mdp = PendulumMDP()

# ╔═╡ 7861a027-c4e2-4b19-9ed9-d71d19d866d6
md"""
**Select $\theta_1$ and $\theta_2$ to maximize the utlity**

theta1 : $(@bind θ1 Slider(-30:0.1:30))
theta2 : $(@bind θ2 Slider(-30:0.1:30))
"""

# ╔═╡ f8023af0-8d13-4d8a-a7d0-a4bff6d3bb1a
θ = [θ1, θ2]

# ╔═╡ 6956a043-b232-4549-a1e0-96625fb0e3ed
policy = FunctionPolicy((s) -> [θ' * s]);

# ╔═╡ b8e76e54-b34c-435e-aec1-36c776d43544
mean_utility = mc_policy_evaluation(mdp, policy)

# ╔═╡ b4f85aef-825d-4961-ac66-3902ade51a41
begin
	animate_pendulum(mdp, policy, "toyexample.gif")
	LocalResource("toyexample.gif")
end

# ╔═╡ e2aa9d1c-c02f-431c-a93a-194c349b8b27
# function expected_return(mdp::PendulumMDP, θ; max_steps=1000, rollouts=100)
# 	policy = FunctionPolicy((s) -> [θ'*s])
# 	sim = RolloutSimulator(max_steps=max_steps)
# 	mean_r = mean([simulate(sim, mdp, policy) for _=1:rollouts])
# 	return mean_r
# end

# ╔═╡ a6a728cf-8ab3-4977-b57b-191118c5564c
md"""
## Local Search (Hooke-Jeeves)

Hooke-Jeeves is an example of a local search method. We saw a local search algorithm in structure learning. In local search, we evaluate the neighbors of a current design point ($\theta$), and move to the neighbor that most improves the value.

The Hooke-Jeeves algorithm is one example of local search. The algorithm takes a step of size ±α in each of the _coordinate directions_ from the current θ. If an improvement is found, it moves to the best point. If no improvement is found, the algorithm shrinks the step size $α$ and repeats. The algorithm terminates once $\alpha$ reaches some minimum value.

The algorithm is illustrated below

$(Resource("https://upload.wikimedia.org/wikipedia/commons/1/14/Direct_search_BROYDEN.gif", :width=>300, :align=>"center"))
"""

# ╔═╡ 77e6cf39-81a6-460e-a3e7-fccd8840df0e
md"""
Now let's code up Hooke-Jeeves and apply it to the inverted pendulum! This code very closely follows the textbook. First, we'll define a special struct to store useful attributes of the algorithm.
"""

# ╔═╡ 1db4fb35-9d5d-4a90-b31a-b8cc71f41a96
struct HookeJeevesPolicySearch
	θ # initial parameterization
	α # step size
	c # step size reduction factor
	ϵ # termination step size
end

# ╔═╡ 5dc60f11-84c8-4a37-90cd-80b0c824348c
md"""
Next, let's define the optimization procedure. We'll take in the algorithm attributes, as well as a function that takes in a parameter vector $θ$ and returns a Monte Carlo estimate of the expected utility. 
"""

# ╔═╡ 9d36dac9-fc4a-433e-9427-6fd9d49c3013
function optimize(M::HookeJeevesPolicySearch, U)
	θ, θ′, α, c, ϵ = copy(M.θ), similar(M.θ), M.α, M.c, M.ϵ
	u = U(θ)
	n = length(θ)
	history = [copy(θ)]
	while α > ϵ
		copyto!(θ′, θ)
		best = (i=0, sgn=0, u=u)
		for i in 1:n
			for sgn in (-1,1)
				θ′[i] = θ[i] + sgn*α
				u′ = U(θ′)
				if u′ > best.u
					best = (i=i, sgn=sgn, u=u′)
				end
			end
			θ′[i] = θ[i]
		end
		if best.i != 0
			θ[best.i] += best.sgn*α
			u = best.u
		else
			α *= c
		end
		push!(history, copy(θ))
	end
	return θ, history
end;

# ╔═╡ c64dc189-784f-45bc-a81f-312099a3a321
md"""
Now we'll define a function our function $U(θ)$
"""

# ╔═╡ 1d900ec9-6d0e-452c-815f-c10da82d1cff
U(θ) = mc_policy_evaluation(mdp, FunctionPolicy((s)->[θ'*s]), max_steps=500, m=10);

# ╔═╡ bb08a157-489a-482f-a080-f957367d945d
md"""
Specify the attributes of our algorithm
"""

# ╔═╡ 5a704698-8793-4fca-9f6e-8c01ef3b9b55
hookejeeves = HookeJeevesPolicySearch(rand(2), 5.0, 0.5, 1e-1);

# ╔═╡ c75f163b-190b-498e-8976-4fd1e0fdc8ea
md"""
Now run the optimization!
"""

# ╔═╡ 55d06d06-2411-438c-b5af-001f2a57d795
md"""
We can also examine the history of points evaluated by the algorithm. The algorithm starts at light blue and progresses toward darker blue.
"""

# ╔═╡ f368e79e-7430-432e-a230-e94e6bece935
md"""
**Q: Suppose we are performing Hooke-Jeeves for policy parameterized by a 3-element vector. We are currently evaluation the point** $[2, -5, 10]$. **Assume the step size is $1$.**

**What points will Hooke-Jeeves evaluate next?**
"""

# ╔═╡ ed482d6b-6779-4a57-aa57-e5e98ded2b5d
# md"""
# A: 

# * Coordinate 1: (3, -5, 10) and (1, -5, 10)
# * Coordinate 2: (2, -4, 10) and (2, -6 10)
# * Coordinate 3 (2, -5, 9) and (2, -5 11)
# """

# ╔═╡ 312b4218-731e-4005-807a-9c78e675cfe2
# md"""
# A: Hooke-Jeeves will evaluate $6$ points
# * Coordinate 1: $[3, -5, 10]$ and $[1, -5, 10]$
# * Coordinate 2: $[2, -4, 10]$ and $[2, -6, 10]$
# * Coordinate 3: $[2, -5, 11]$ and $[2, -5, 9]$
# """

# ╔═╡ e0ccd05a-c730-443d-b3d7-7c38b5c38a90


# ╔═╡ db8209b7-8fa6-4bb4-bfe3-680aa92e14d1
md"""
**Q:Hooke-Jeeves evaluates each of these points, and finds they have utilities** 

$(10, 23, 16, 5, 34, 27)$.
**The current policy has a value of $35$. What will Hooke-Jeeves do?**
"""

# ╔═╡ c7850642-77aa-42c3-9bf9-dde155050baf
# md"""
# A: Shrink!
# """

# ╔═╡ d1b35fc9-93a8-4f9e-8605-78907c87b973
# md"""
# A: Hooke-Jeeves will contract the step size. The utility at each neighbor is lower than the current policy.
# """

# ╔═╡ a362110b-dfa2-4a92-aab0-a22ff463b9ef
md"""
## Genetic Algorithms

Local search algorithms can easily become stuck in local minima. Population-based algorithms maintain a set of points in the parameter space. By maintaining and changing the set of parameters, population-based algorithms can be less susceptible to becoming stuck. However, they are not guaranteed to converge to the global optimum.

Genetic algorithms are population-based algorithms that are inspired by biological evolution. The algorithm starts with a population of points $m$ in parameter space, called 'individuals': $\theta^{(1)},\ldots,\theta^{(m)}$. We compute $U(\theta)$ for each of point. The top-performing samples, are called _elite samples_. At the next iteration, some number $m_{elite}$ of the elite samples are chosen. New points in the population are created by adding gaussian noise to the elite individuals.
"""

# ╔═╡ a14e488d-4638-496a-8a60-6a51ba86915f
struct GeneticPolicySearch
	θs # initial population
	σ # initial standard deviation
	m_elite # number of elite samples
	k_max # number of iterations
end

# ╔═╡ e41ef855-c3c8-45f8-b543-664dbfa38c1a
function optimize(M::GeneticPolicySearch, U)
	θs, σ = M.θs, M.σ
	n, m = length(first(θs)), length(θs)
	history = []
	for k in 1:M.k_max
		us = [U(θ) for θ in θs]
		sp = sortperm(us, rev=true)
		θ_best = θs[sp[1]]
		push!(history, (copy(θs), copy(θs[sp[1:M.m_elite]])))
		rand_elite() = θs[sp[rand(1:M.m_elite)]]
		θs = [rand_elite() + σ.*randn(n) for i in 1:(m-1)]
		push!(θs, θ_best)
	end
	return last(θs), history
end;

# ╔═╡ 6e87fd3c-7de9-496d-a2f9-92bd35dddd70
md"""
Creating an initial population
"""

# ╔═╡ 8e8481bc-b8a9-4467-b05e-26f946d6f805
begin
	npop = 50
	θ0 = [50 .* rand(2).-25 for _=1:npop]
end

# ╔═╡ 687b6b83-d2b3-43e3-b480-6adb323f0a71
ga = GeneticPolicySearch(θ0, 1.0, 10, 10);

# ╔═╡ 8bc6acb1-c986-4ff2-9b86-f7f9f3091fb2
md"""
The final parameters found using the genetic algorithm are:
"""

# ╔═╡ 41dca1da-ee51-491b-af36-ccab85b76adf
md"""
With an expected utility of:
"""

# ╔═╡ d1e4f0b9-6354-40ca-97c2-0145a6652203
md"""
Let's take a look a the population over each iteration. Try playing with the initial population, standard deviation, and other parameters and see how the algorithm behaves. 
"""

# ╔═╡ 112b0d73-284b-443d-b0e3-b61539832020
@bind igen Slider(1:(ga.k_max))

# ╔═╡ fb8552f4-39a6-4bdc-ab41-47047eb6563a
md"""
**Q: How will the solution found by a genetic algorithm depend on the initial population? How would it depend on the magnitude of noise added to elite samples?**
"""

# ╔═╡ 083af011-1f87-4370-b059-74411984247a
# md"""
# **A:** Generally, a more distributed population can more effectively explore the parameter space at the cost of computation (need for policy evaluation for more individuals) and convergence speed.

# There is a similar story for the magnitude for the random perturbations. Adding random noise helps explore the policy space, but can also slow down convergence.
# """

# ╔═╡ 3aead6fc-4b6a-482a-bb37-7abe4b306c70


# ╔═╡ cb0fbc5f-3e50-45fc-b821-f27fa999bfa4
md"""
**Q: Is a genetic algorithm guaranteed to converge to the optimal policy?**
"""

# ╔═╡ c0a9dedc-aa3f-4315-ac4d-cdf6203d8be5
# md"""
# **A:** No, it is not guaranteed to converge to optimal policy. The addition of random perturbations in the policy parameters means we may never converge to an optimum.
# """

# ╔═╡ 07eb9cb6-83ac-41f3-96da-9ca97441eb90


# ╔═╡ 3c633af9-a03f-4a8e-97e6-fddc39379b45
md"""
## Cross Entropy Method

The cross entropy method involves updating a search distribution over the parameters. The distribution over parameters $p(\theta \mid \psi)$ has its own parameters $\psi$. Typically, we use a Gaussian distribution, where $\psi$ represents the mean and covariance matrix.

The algorithm iteratively updates the parameters $\psi$. At each iteration, we draw $m$ samples from the associated distribution and then update ψ to fit a set of elite samples. We stop after a fixed number of iterations, or when the search distribution becomes very focused.
"""

# ╔═╡ bcb12c46-798e-4c7e-98e6-d17c39973a0d
struct CrossEntropyPolicySearch
	p # initial distribution
	m # number of samples
	m_elite # number of elite samples
	k_max # number of iterations
end

# ╔═╡ aa837034-8705-4407-8233-32bf39b78ed6
function optimize_dist(M::CrossEntropyPolicySearch, U)
	p, m, m_elite, k_max = M.p, M.m, M.m_elite, M.k_max
	history = []
	for k in 1:k_max
		θs = rand(p, m)
		us = [U(θs[:,i]) for i in 1:m]
		θ_elite = θs[:,sortperm(us)[(m-m_elite+1):m]]
		push!(history, (p, copy(θs), copy(θ_elite)))
		p = Distributions.fit(typeof(p), θ_elite)
		
	end
	return p, history
end;

# ╔═╡ 0a203e2b-d1a0-4258-9f08-7f6674f62e68
function optimize(M, U)
	d, history = optimize_dist(M, U)	
	return Distributions.mode(d), history
end;

# ╔═╡ a9d4fde4-2bbd-424d-a182-d85b69dc2582
θhj, history = optimize(hookejeeves, U);

# ╔═╡ 5f2a1567-5d5b-411f-a35f-50f56b8ddc60
θhj

# ╔═╡ 96f620a7-a682-4ca5-9385-5b79a23f2bbd
U(θhj)

# ╔═╡ 070934af-41be-4871-a9b8-04f966c82676
begin
	πcem = FunctionPolicy((s)->[θhj'*s])
	animate_pendulum(mdp, πcem, "hj.gif")
	LocalResource("hj.gif")
end

# ╔═╡ 08df5639-7366-442b-a14e-03337663e1d7
plot(map(t->t[1], history), map(t->t[2], history), legend=nothing, markershape=:circle, markersize=6, linewidth=5, marmarker_z=1:length(history), color=colormap("Blues", length(history)), xlabel="θ1", ylabel="θ2", aspect_ratio=1, left_margin=20px)

# ╔═╡ 805dd1d1-303d-4888-9fbc-7016bb362394
θga, hga = optimize(ga, U);

# ╔═╡ 26b2ef30-84a2-4010-9c77-682e64340533
θga

# ╔═╡ 341e0f53-0811-4da7-ab00-93b97b24596f
U(θga)

# ╔═╡ f4626fb9-9e7a-43ae-819b-2e78cfd2371e
begin
	xga = map(t->t[1], hga[igen][1])
	yga = map(t->t[2], hga[igen][1])
	xge = map(t->t[1], hga[igen][2])
	yge = map(t->t[2], hga[igen][2])
	plot(xga, yga, seriestype=:scatter, xlabel="θ1", ylabel="θ2", xlims=[-30, 30], ylims=[-30, 30], title="Generation $igen", label=nothing)
	plot!(xge, yge, seriestype=:scatter, c=:red, label="Elite")
end

# ╔═╡ ed3beaf2-bbdb-4da4-bf1f-0910b5690918
md"""
A key step of using the algorithm is selecting the _initial distribution_. The distribution should cover the parameter space of interest.
"""

# ╔═╡ 69b8ca8a-ae7c-4824-8002-bb45b09c9ded
p0 = MvNormal(zeros(2), [5, 5])

# ╔═╡ c81e1340-e21e-4e28-b55f-c629986ef3c6
cem = CrossEntropyPolicySearch(p0, 50, 10, 10)

# ╔═╡ c9957bc6-df4f-4727-8730-9d452b7b1064
θcem, hcem = optimize(cem, U);

# ╔═╡ e4882406-d5d4-4c0a-979d-7cc5cbb2159f
θcem

# ╔═╡ 597199d5-284d-4ca9-8dd8-73d573013704
U(θcem)

# ╔═╡ 78a7d414-3dd5-44e3-9038-bceb21b155d0
md"""
We can also examine how the proposal distribution changes over iterations.
"""

# ╔═╡ 2f98fd42-a5cf-4431-be24-58d1a1b5bdaa
@bind icem Slider(1:(cem.k_max))

# ╔═╡ 4d198787-c6e1-49f1-9c45-16ca2684e4be
begin
	cem_samples = hcem[icem][2]
	cem_p = hcem[icem][1]
	xcem = cem_samples[1, :]
	ycem = cem_samples[2, :]
	xelite = hcem[icem][3][1, :]
	yelite = hcem[icem][3][2, :]
	plot(xcem, ycem, seriestype=:scatter, xlabel="θ1", ylabel="θ2", xlims=[-30, 30], ylims=[-30, 30], title="CEM Iteration $icem", label=nothing, aspect_ratio=1)
	plot!(xelite, yelite, seriestype=:scatter, label="Elite Points", c=:red)
	covellipse!(cem_p.μ, cem_p.Σ, n_std=2, label=nothing, fillcolor=:transparent, linecolor=:black, linewidth=2, fillalpha=0, markeralpha=1, linealpha=1)
end

# ╔═╡ db4b3c21-60ac-43f5-aa77-f6f0654dbc41
md"""
**Q: We are performing the Cross Entropy Method on a 1D parameter space. We have elite samples at $(1, 4, 2.5, 10)$ What will be the updated parameters for our Gaussian search distribution?**

Recall the maximum likelihood estimate for Gaussian's parameters with samples $o_1,\ldots,o_m$:

$\hat{\mu} = \frac{\sum_i o_i}{m}$
$\hat{\sigma}^2 = \frac{\sum_i (o_i - \hat{\mu})^2}{m}$
"""

# ╔═╡ 1afa7067-dda8-4509-a5ee-d219e46796bd
# md"""
# A: 
# """

# ╔═╡ 1160e802-aaa3-46c9-9bfa-8c9eca34c9f4
# md"""
# A: We use a maximum likelihood estimate to update the paramters. For a Gaussian distribution and samples $o_1$

# The new **mean** is (1+4+2.5+10)/4= $((1+4+2.5+10)/4)

# The new **variance** is ((1-4.375)^2 + (4-4.375)^2 + (2.5-4.375)^2 + (10-4.375)^2)/4 = $(((1-4.375)^2 + (4-4.375)^2 + (2.5-4.375)^2 + (10-4.375)^2)/4)
# """

# ╔═╡ 133e8543-b099-447d-b1bb-4356f4eb7809


# ╔═╡ ad95db56-6c0e-4415-8382-afc80b183223
md"""
## Algorithm Comparison

Let's compare the performance of each algorithm on the inverted pendulum
"""

# ╔═╡ 5a8975d6-78c2-4392-8a15-6ffc0cb49bf0
begin
	@show U(θhj)
	@show U(θga)
	@show U(θcem)
end;

# ╔═╡ 390d4ef0-e85f-4184-a3cb-64733f97d176
md"""
They all do pretty well!
"""

# ╔═╡ be6270b4-930a-4f8e-b7f1-e0a253a0ae9f
md"""
**Q:What if our policy had many more parameters, say 100. Which algorithm would you pick?**
"""

# ╔═╡ 26e2c454-4f07-4146-b5da-521d7ccd7c39
# md"""
# A:
# """

# ╔═╡ 13acb582-9eac-4c20-bd0e-457fa0340500


# ╔═╡ f706733d-2460-41a6-a45f-c07d5b4b537a
md"""
**Q: Suppose that we want to perform policy optimization on a problem where we know that policies far apart in parameter space can have similar high utility. What are the advantages of genetic algorithms over hooke-jeeves? What about the Cross Entropy method?**
"""

# ╔═╡ 5f9e1c12-fb9c-4b24-bf63-cb4250585814
# md"""
# A:
# """

# ╔═╡ f845c0ef-7710-4341-b298-1d04892032f5


# ╔═╡ 120af26f-23dc-43ed-8f1c-3600c7bc2878
md"""
## Next Steps: Gradient Information

Thus far, all of the algorithms for policy search have not used gradient information of the expected utility with respect to policy parameters. It turns out that optimizing policies with many parameters can be done much more efficiently with gradient information

How can we compute the gradient $\nabla U(\theta)$?

One option is to use _finite differences_. The idea of finite differences comes from the linear approximation of the gradient. We can estimate the gradient (or the slope) of a function in 1D by checking how much the value of $f(x)$ changes for some small change in $x$.

$\frac{df}{dx} \approx \frac{f(x + \delta) - f(x)}{\delta}$

If we extend this same idea to our utility function,

$\nabla U(\theta) \approx \Big[ \frac{U(\theta + \delta \mathbf{e}^1) - U(\theta)}{\delta}, \ldots, \frac{U(\theta + \delta \mathbf{e}^n) - U(\theta)}{\delta}  \Big]$

Where $\mathbf{e}^i$ is the standard basis, which is zero everywhere except the $i$th component.

Luckily, there are some great packages in most programming languages that do this for us!

However, recall that our Monte Carlo estimate of the expected utility is stochastic, or noisy. This means that gradients of the utility function will have noise too! If the gradients are too noisy, they will provide very poor guidance for our policy search.

A key challenge in policy gradient estimation is dealing with noisy policy gradients.
"""

# ╔═╡ 716832f9-1180-4fba-9e01-bda3057eb206
function deterministic_policy_evaluation(mdp::MDP, π::Policy; m=100, max_steps=100)
	sim = RolloutSimulator(rng=MersenneTwister(42), max_steps=max_steps)
	return mean([simulate(sim, mdp, π) for _=1:m])
end;

# ╔═╡ 8e4d6bad-32e0-4bef-a55b-a13d6f5999fe
Ufd(θ) = deterministic_policy_evaluation(mdp, FunctionPolicy((s)->[θ'*s]), max_steps=100, m=5);

# ╔═╡ 791912c4-fc1c-4f1a-be80-56f4d939e472
md"""
Let's try taking the policy gradient.
"""

# ╔═╡ 2449fd2d-3e98-4133-aa9d-c735b4348912
FiniteDiff.finite_difference_gradient(Ufd, [0.1, 0.1])

# ╔═╡ 92d6333d-b605-4308-aeb2-d69e6da9ff9d
md"""
Now that we have an estimate of the gradient, how can we use it to improve the policy?

One of the simplest approaches is **gradient ascent**. Gradient ascent takes steps in parameter space along the gradient direction. The step size $\alpha$ determines how far along the gradient direction the update moves. The update for $θ$ is

$\theta \leftarrow \theta + α \nabla U(θ)$

Determining the step size is a major challenge. Large steps can lead to faster progress to the optimum, but they can overshoot. 

Let's try running a very simple version of gradient descent and see how it performs!
"""

# ╔═╡ c0756af5-4951-4d1a-9af5-25835794d944
begin
	θi = rand(2)
	α = 1e-1
	niter = 500
	for k=1:niter
		gradU = FiniteDiff.finite_difference_gradient(Ufd, θi)
		θi += α .* gradU
	end
end

# ╔═╡ 04586b86-a14b-44f1-bb84-e591912f7d98
md"""
The final parameters are a little different than what was previously found. Why could that be?
"""

# ╔═╡ 1bc8ba36-f1f0-4645-b87c-a4997452c907
θi

# ╔═╡ 171f4942-a81e-4a7f-b0b7-61338b74e33c
Ufd(θi)

# ╔═╡ 19ae5fe0-e5ee-4040-ab71-fd9ca4fd1aae
begin
	πsgd = FunctionPolicy((s)->[θi'*s])
	animate_pendulum(mdp, πsgd, "sgd.gif")
	LocalResource("sgd.gif")
end

# ╔═╡ fb271fd9-3b7f-431b-9830-5282e9792e52
md"""
Gradient descent is able to find a decent policy pretty quickly! However, it isn't as good as the policies we've found previously. It turns out there are **much** better ways to estimate the gradient of a policy, and much more intelligent variations on gradient ascent. We'll explore these in the coming lecture.
"""

# ╔═╡ e00c7040-fc5e-4e92-922b-800ee0f13a42
md"""
This wraps up our discussion on policy search. I hope it was helpful!
"""

# ╔═╡ 91472d8c-2e4c-4ddd-a925-9d691a854239
ReLU(z) = max(z, 0);

# ╔═╡ d52f005e-200c-40c4-8745-7119d1774bbc
function neural_network(x, 𝐕, 𝐰, φ, g=ReLU)
    𝐡 = ReLU.(𝐕 * φ(x))
    𝐰 ⋅ 𝐡
end;

# ╔═╡ b936727f-2d87-49b1-b8fe-4b9150fc05e1
function pendulum_nn(s, θ)
	ϕ(s) = [cos(s[1]), sin(s[1]), s[2]]
	V = reshape(θ[1:(4*3)], 4, 3)
	w = reshape(θ[13:end], 1, 4)
	return neural_network(s, V, w, ϕ)
end;

# ╔═╡ b7f2af2d-3c59-4fb4-a166-bdae243b7d0f
s = [0.5, 0.1];

# ╔═╡ 172d2506-60b7-4da2-a433-7318425ddc08
θnn = rand(16);

# ╔═╡ 09930b6c-9c78-4c74-b4bb-6064745838c8
pendulum_nn(s, θnn);

# ╔═╡ 69039968-3041-4450-855e-df23dc23218e
Unn(θ) = mc_policy_evaluation(mdp, FunctionPolicy((s)->[pendulum_nn(s, θ)]), max_steps=500, m=10);

# ╔═╡ 1d1e0228-f3e9-4b2b-9d7e-b5839cbc929a
hjnn = HookeJeevesPolicySearch(zeros(16), 5.0, 0.5, 1e-1);

# ╔═╡ 412cc766-f734-4578-8169-11e185cf69c4
optimize(hjnn, Unn);

# ╔═╡ 4a23f7c5-2365-4852-8c83-4e523a6780a5
begin
	nnpop = 100
	θ0nn = [2 .* rand(16) for _=1:nnpop]
end;

# ╔═╡ 6772601e-351c-4836-921e-91467b82e29b
gann = GeneticPolicySearch(θ0nn, 5.0, 10, 10);

# ╔═╡ 774660b0-a984-469d-b2b1-0998a1464df6
#θnn_ga = optimize(gann, Unn)

# ╔═╡ d771cbaf-7025-436b-a231-5af79ebfd5b3
#Unn(θnn_ga)

# ╔═╡ 48f999f0-0c0d-4fb9-babb-01bd87673530
#cemnn = CrossEntropyPolicySearch(MvNormal(zeros(16), 2 .* ones(16)), 100, 10, 10)

# ╔═╡ ad12e8cb-9bf2-4947-bb64-70179166a722
#θnncem = optimize(cemnn, Unn)

# ╔═╡ 4c5e90ec-c696-45d5-866d-c8568933a876
#Unn(θnncem)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Cairo = "159f3aea-2a34-519c-b102-8c37f9878175"
Compose = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
ImageCore = "a09fc81d-aa75-5fe9-8630-4744c3626534"
ImageMagick = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
ImageShow = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
POMDPTools = "7588e00f-9cae-40de-98dc-e0c70c48cdd7"
POMDPs = "a93abf59-7444-517b-a68a-c42f96afdd7d"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Cairo = "~1.0.5"
Compose = "~0.9.4"
Distributions = "~0.25.80"
FiniteDiff = "~2.17.0"
ImageCore = "~0.9.4"
ImageMagick = "~1.2.2"
ImageShow = "~0.3.6"
Images = "~0.25.2"
POMDPTools = "~0.1.2"
POMDPs = "~0.9.5"
Parameters = "~0.12.3"
Plots = "~1.38.4"
PlutoUI = "~0.7.49"
StatsPlots = "~0.15.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "13927749069d5a4a7358c4d160c20c03914b96ec"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "0310e08cb19f5da31d08341c6120c047598f5b9c"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e5f08b5689b1aad068e01751889f2f615c7db36d"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.29"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "1dd4d9f5beebac0c03446918741b1a03dc5e5788"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.6"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c6d890a52d2c4d55d326439580c3b8d0875a77d9"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.7"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "844b061c104c408b24537482469400af6075aae4"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.5"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "64df3da1d2a26f4de23871cd1b6482bb68092bd5"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.3"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random", "SnoopPrecompile"]
git-tree-sha1 = "aa3edc8f8dea6cbfa176ee12f7c2fc82f0608ed3"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.20.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonRLInterface]]
deps = ["MacroTools"]
git-tree-sha1 = "21de56ebf28c262651e682f7fe614d44623dc087"
uuid = "d842c3ba-07a1-494f-bbec-f5741b0a3e98"
version = "0.3.1"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "00a2cccc7f098ff3b66806862d275ca3db9e6e5a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.5.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Compose]]
deps = ["Base64", "Colors", "DataStructures", "Dates", "IterTools", "JSON", "LinearAlgebra", "Measures", "Printf", "Random", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "d853e57661ba3a57abcdaa201f4c9917a93487a2"
uuid = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
version = "0.9.4"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fb21ddd70a051d882a1686a5a550990bbe371a95"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.1"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "681ea870b918e7cff7111da58791d7f718067a19"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d4f69885afa5e6149d0cab3818491565cf41446d"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.4.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "74911ad88921455c6afcad1eefa12bd7b1724631"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.80"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "7be5f99f7d15578798f338f5433b6c432ea8037b"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "d3ba08ab64bdfd27234d3f61956c966266757fe6"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.7"

[[deps.FiniteDiff]]
deps = ["ArrayInterfaceCore", "LinearAlgebra", "Requires", "Setfield", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "04ed1f0029b6b3af88343e439b995141cb0d0b8d"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.17.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "cabd77ab6a6fdff49bfd24af2ebe76e6e018a2b4"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.0.0"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "9e23bd6bb3eb4300cb567bdf63e2c14e5d2ffdbc"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.71.5"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "aa23c9f9b7c0ba6baeabe966ea1c7d2c7487ef90"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.71.5+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "ba2d094a88b6b287bd25cfa86f301e7693ffae2f"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.7.4"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "37e4657cd56b11abe3d10cd4a1ec5fbdb4180263"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.7.4"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "c54b581a83008dc7f292e205f4c409ab5caa0f04"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.10"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageContrastAdjustment]]
deps = ["ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "0d75cafa80cf22026cea21a8e6cf965295003edc"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.10"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "b1798a4a6b9aafb530f8f0c4a7b2eb5501e2f2a3"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.16"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "Reexport", "SnoopPrecompile", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "f265e53558fbbf23e0d54e4fab7106c0f2a9e576"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.3"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "342f789fd041a55166764c351da1710db97ce0e0"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.6"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils"]
git-tree-sha1 = "ca8d917903e7a1126b6583a097c5cb7a0bedeac1"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.2"

[[deps.ImageMagick_jll]]
deps = ["JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1c0a2295cca535fabaf2029062912591e9b61987"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.10-12+3"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "36cbaebed194b292590cba2593da27b34763804a"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.8"

[[deps.ImageMorphology]]
deps = ["ImageCore", "LinearAlgebra", "Requires", "TiledIteration"]
git-tree-sha1 = "e7c68ab3df4a75511ba33fc5d8d9098007b579a8"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.3.2"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "5985d467623f106523ed8351f255642b5141e7be"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.4"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "36832067ea220818d105d718527d6ed02385bf22"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.7.0"

[[deps.ImageShow]]
deps = ["Base64", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "b563cf9ae75a635592fc73d3eb78b86220e55bd8"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.6"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "8717482f4a2108c9358e5c3ca903d3a6113badc9"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.9.5"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "03d1301b7ec885b266c0f816f338368c6c0b81bd"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.25.2"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "be8e690c3973443bec584db3346ddc904d4884eb"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "16c0cc91853084cb5f58a78bd209513900206ce6"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.4"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.InvertedIndices]]
git-tree-sha1 = "82aec7a3dd64f4d9584659dc0b62ef7db2ef3e19"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.2.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "c3244ef42b7d4508c638339df1bdbf4353e144db"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.30"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "a77b273f1ddec645d1b7c4fd5fb98c8f90ad10a5"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "9816b296736292a80b9a3200eb7fbb57aaa3917a"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.5"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "2422f47b34d4b127720a18f86fa7b1aa2e141f29"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.18"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "45b288af6956e67e621c5cbb2d75a261ab58300b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.20"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[deps.MarchingCubes]]
deps = ["SnoopPrecompile", "StaticArrays"]
git-tree-sha1 = "3738199de01df8fec0a9b0d96fd311aac71fe6f6"
uuid = "299715c1-40a9-479a-aaf9-4a633d36f717"
version = "0.1.5"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "1130dbe1d5276cb656f6e1094ce97466ed700e5a"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "efe9c8ecab7a6311d4b91568bd6c88897822fabe"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NamedTupleTools]]
git-tree-sha1 = "90914795fc59df44120fe3fff6742bb0d7adb1d0"
uuid = "d9ec5142-1e00-5aa0-9d6a-321866360f50"
version = "0.14.3"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "5ae7ca23e13855b3aba94550f26146c01d259267"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "f71d8950b724e9ff6110fc948dff5a329f901d64"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.8"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "6503b77492fd7fcb9379bf73cd31035670e3c509"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.3.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6e9dba33f9f2c44e08a020b0caf6903be540004"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.19+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "f809158b27eba0c18c269cf2a2be6ed751d3e81d"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.17"

[[deps.POMDPLinter]]
deps = ["Logging"]
git-tree-sha1 = "cee5817d06f5e1a9054f3e1bbb50cbabae4cd5a5"
uuid = "f3bd98c0-eb40-45e2-9eb1-f2763262d755"
version = "0.1.1"

[[deps.POMDPTools]]
deps = ["CommonRLInterface", "DataFrames", "Distributed", "Distributions", "LinearAlgebra", "NamedTupleTools", "POMDPLinter", "POMDPs", "Parameters", "ProgressMeter", "Random", "Reexport", "SparseArrays", "Statistics", "StatsBase", "Tricks", "UnicodePlots"]
git-tree-sha1 = "ef73c26402974cd51b9bd395ba5d95a4e34c1b37"
uuid = "7588e00f-9cae-40de-98dc-e0c70c48cdd7"
version = "0.1.2"

[[deps.POMDPs]]
deps = ["Distributions", "Graphs", "NamedTupleTools", "POMDPLinter", "Pkg", "Random", "Statistics"]
git-tree-sha1 = "9ab2df9294d0b23def1e5274a0ebf691adc8f782"
uuid = "a93abf59-7444-517b-a68a-c42f96afdd7d"
version = "0.9.5"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "03a7a85b76381a3d04c7a1656039197e70eda03d"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.11"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "84a314e3926ba9ec66ac097e3635e270986b0f10"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.50.9+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "8175fc2b118a3755113c8e68084dc1a9e63c61ee"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.3"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f6cf8e7944e50901594838951729a1861e668cb8"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.2"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "c95373e73290cf50a8a22c3375e4625ded5c5280"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "87036ff7d1277aa624ce4d211ddd8720116f80bf"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.4"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eadad7b14cf046de6eb41f13c9275e5aa2711ab6"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.49"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "96f6db03ab535bdb901300f88335257b0018689d"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "de191bc385072cc6c7ed3ffdc1caeed3f22c74d4"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.7.0"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "da095158bdc8eaccb7890f9884048555ab771019"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "261dddd3b862bd2c940cf6ca4d1c8fe593e457c8"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.3"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase", "SnoopPrecompile"]
git-tree-sha1 = "e974477be88cb5e3040009f3767611bc6357846f"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.11"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "9480500060044fd25a1c341da53f34df7443c2f2"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.3.4"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "c02bd3c9c3fc8463d3591a62a378f90d2d8ab0f3"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.17"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays", "Test"]
git-tree-sha1 = "a8d28ad975506694d59ac2f351e29243065c5c52"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.2.2"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "8fb59825be681d451c246a795117f317ecbcaa28"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.2"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "6954a456979f23d05085727adb17c4551c19ecd1"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.12"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "ab6083f09b3e617e34a956b43e9d51b824206932"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.1.1"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "e0d5bc26226ab1b7648278169858adcfbd861780"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.4"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "7e6b0e3e571be0b4dd4d2a9a3a83b65c04351ccc"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.3"

[[deps.TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "5683455224ba92ef59db72d10690690f4a8dc297"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.1"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "94f38103c984f89cf77c402f2a68dbd870f8165f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.11"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "ac00576f90d8a259f2c9d823e91d1de3fd44d348"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnicodePlots]]
deps = ["ColorSchemes", "ColorTypes", "Contour", "Crayons", "Dates", "FileIO", "FreeType", "LinearAlgebra", "MarchingCubes", "NaNMath", "Printf", "Requires", "SnoopPrecompile", "SparseArrays", "StaticArrays", "StatsBase", "Unitful"]
git-tree-sha1 = "64b6f4f8ffd783af63819126cb3091f97c3b1aec"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "3.3.4"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d670a70dd3cdbe1c1186f2f17c9a68a7ec24838c"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.12.2"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─e82b0afc-4072-4ae5-aca3-dd0542165469
# ╟─a7fb576c-afc5-4be2-a5be-b722b312f6d0
# ╟─18d9bf07-2a48-4bac-8a72-c46b5a0e7baf
# ╟─48b00590-f5c2-4ec1-95d3-7da1dd861e88
# ╟─0cd75ebe-a86a-471b-8b13-2180f0bd25fa
# ╟─1fce7b86-8413-499d-ae3a-f2a95496a0e2
# ╟─24a02196-09b0-4249-8a5b-42b3217e8ad9
# ╟─76a5ba49-eff1-4aff-80eb-156963702404
# ╠═b2cf3f80-9e12-11ed-2cd7-fddcd14709e3
# ╟─49198ec3-6f5f-43a9-af14-eaae60e81142
# ╟─f930bfce-0810-4748-82a9-9004176be619
# ╟─85ecd311-6760-4b65-ad98-6f0d69eab436
# ╟─5b94e6b1-b6ef-4428-9f24-c061264a8cee
# ╟─e35204bd-ca5e-45c0-94b5-0507575c9984
# ╟─28f3b33c-1298-4d5b-8bbc-5a7af55e5b1e
# ╟─6ccd55cb-13f0-4ab6-a2f8-d12eae87e159
# ╟─4dc17f13-c1fa-43c5-a5aa-0c13f4062ed7
# ╠═2d181244-cfe5-4158-880d-b920b14320db
# ╠═299c6844-27d6-4488-b18d-1f0b8796b025
# ╟─6b3385a3-418c-491d-9bff-bf4dc9c2ff5d
# ╠═59dc4f6a-0633-4a14-acef-4f6083ccd058
# ╠═f05e13d5-9e91-420c-8f38-72509d5a6723
# ╠═021b3e10-1d4c-4e08-8d09-5556103ebb46
# ╟─9fb8392c-ae88-4829-8525-20d920e8c6b5
# ╠═992f0385-8d35-49c9-ab8b-14bd8e33b4ef
# ╟─a2784cb8-c534-4c41-8254-342f5eb16b9f
# ╟─831a51e7-cd0e-4216-87df-263cdf7acc24
# ╟─5342b6a4-f8ba-46be-baaa-be2ef3b4b9ec
# ╠═2b045ae4-1176-44db-be0b-042f788b8e2c
# ╟─74a1237d-0d70-4686-b0fd-d3b4e41be2d7
# ╟─cc29ce34-7719-404a-81c8-22af6e44b680
# ╟─5650c7cd-a977-4d4d-b63e-ba8db23bcefc
# ╟─82804c10-2a66-41a8-9eec-96282588c386
# ╟─aa1265ae-1b99-41ef-bce4-f24c946d066f
# ╟─00aedb7b-c0df-4344-acc8-2bb1cdc83db6
# ╟─a4e51e09-6288-425a-82af-f6dd4e019d1b
# ╠═b0f8a206-ee75-4d6c-bd19-8ae840df46b2
# ╟─33a1f2a2-188a-4674-b49d-0346f23449e8
# ╠═50018cd7-8427-44fd-aeb2-dc25853d912c
# ╟─7861a027-c4e2-4b19-9ed9-d71d19d866d6
# ╠═f8023af0-8d13-4d8a-a7d0-a4bff6d3bb1a
# ╠═6956a043-b232-4549-a1e0-96625fb0e3ed
# ╠═b8e76e54-b34c-435e-aec1-36c776d43544
# ╟─b4f85aef-825d-4961-ac66-3902ade51a41
# ╟─e2aa9d1c-c02f-431c-a93a-194c349b8b27
# ╟─a6a728cf-8ab3-4977-b57b-191118c5564c
# ╟─77e6cf39-81a6-460e-a3e7-fccd8840df0e
# ╠═1db4fb35-9d5d-4a90-b31a-b8cc71f41a96
# ╟─5dc60f11-84c8-4a37-90cd-80b0c824348c
# ╠═9d36dac9-fc4a-433e-9427-6fd9d49c3013
# ╟─c64dc189-784f-45bc-a81f-312099a3a321
# ╠═1d900ec9-6d0e-452c-815f-c10da82d1cff
# ╟─bb08a157-489a-482f-a080-f957367d945d
# ╠═5a704698-8793-4fca-9f6e-8c01ef3b9b55
# ╟─c75f163b-190b-498e-8976-4fd1e0fdc8ea
# ╠═a9d4fde4-2bbd-424d-a182-d85b69dc2582
# ╠═5f2a1567-5d5b-411f-a35f-50f56b8ddc60
# ╠═96f620a7-a682-4ca5-9385-5b79a23f2bbd
# ╟─070934af-41be-4871-a9b8-04f966c82676
# ╟─55d06d06-2411-438c-b5af-001f2a57d795
# ╟─08df5639-7366-442b-a14e-03337663e1d7
# ╟─f368e79e-7430-432e-a230-e94e6bece935
# ╟─ed482d6b-6779-4a57-aa57-e5e98ded2b5d
# ╟─312b4218-731e-4005-807a-9c78e675cfe2
# ╟─e0ccd05a-c730-443d-b3d7-7c38b5c38a90
# ╟─db8209b7-8fa6-4bb4-bfe3-680aa92e14d1
# ╟─c7850642-77aa-42c3-9bf9-dde155050baf
# ╟─d1b35fc9-93a8-4f9e-8605-78907c87b973
# ╟─a362110b-dfa2-4a92-aab0-a22ff463b9ef
# ╠═a14e488d-4638-496a-8a60-6a51ba86915f
# ╠═e41ef855-c3c8-45f8-b543-664dbfa38c1a
# ╟─6e87fd3c-7de9-496d-a2f9-92bd35dddd70
# ╠═8e8481bc-b8a9-4467-b05e-26f946d6f805
# ╠═687b6b83-d2b3-43e3-b480-6adb323f0a71
# ╠═805dd1d1-303d-4888-9fbc-7016bb362394
# ╟─8bc6acb1-c986-4ff2-9b86-f7f9f3091fb2
# ╠═26b2ef30-84a2-4010-9c77-682e64340533
# ╟─41dca1da-ee51-491b-af36-ccab85b76adf
# ╠═341e0f53-0811-4da7-ab00-93b97b24596f
# ╟─d1e4f0b9-6354-40ca-97c2-0145a6652203
# ╟─112b0d73-284b-443d-b0e3-b61539832020
# ╟─f4626fb9-9e7a-43ae-819b-2e78cfd2371e
# ╟─fb8552f4-39a6-4bdc-ab41-47047eb6563a
# ╟─083af011-1f87-4370-b059-74411984247a
# ╟─3aead6fc-4b6a-482a-bb37-7abe4b306c70
# ╟─cb0fbc5f-3e50-45fc-b821-f27fa999bfa4
# ╟─c0a9dedc-aa3f-4315-ac4d-cdf6203d8be5
# ╟─07eb9cb6-83ac-41f3-96da-9ca97441eb90
# ╟─3c633af9-a03f-4a8e-97e6-fddc39379b45
# ╠═bcb12c46-798e-4c7e-98e6-d17c39973a0d
# ╠═aa837034-8705-4407-8233-32bf39b78ed6
# ╠═0a203e2b-d1a0-4258-9f08-7f6674f62e68
# ╟─ed3beaf2-bbdb-4da4-bf1f-0910b5690918
# ╠═69b8ca8a-ae7c-4824-8002-bb45b09c9ded
# ╠═c81e1340-e21e-4e28-b55f-c629986ef3c6
# ╠═c9957bc6-df4f-4727-8730-9d452b7b1064
# ╠═e4882406-d5d4-4c0a-979d-7cc5cbb2159f
# ╠═597199d5-284d-4ca9-8dd8-73d573013704
# ╟─78a7d414-3dd5-44e3-9038-bceb21b155d0
# ╟─2f98fd42-a5cf-4431-be24-58d1a1b5bdaa
# ╟─4d198787-c6e1-49f1-9c45-16ca2684e4be
# ╟─db4b3c21-60ac-43f5-aa77-f6f0654dbc41
# ╟─1afa7067-dda8-4509-a5ee-d219e46796bd
# ╟─1160e802-aaa3-46c9-9bfa-8c9eca34c9f4
# ╟─133e8543-b099-447d-b1bb-4356f4eb7809
# ╟─ad95db56-6c0e-4415-8382-afc80b183223
# ╠═5a8975d6-78c2-4392-8a15-6ffc0cb49bf0
# ╟─390d4ef0-e85f-4184-a3cb-64733f97d176
# ╟─be6270b4-930a-4f8e-b7f1-e0a253a0ae9f
# ╟─26e2c454-4f07-4146-b5da-521d7ccd7c39
# ╟─13acb582-9eac-4c20-bd0e-457fa0340500
# ╟─f706733d-2460-41a6-a45f-c07d5b4b537a
# ╟─5f9e1c12-fb9c-4b24-bf63-cb4250585814
# ╟─f845c0ef-7710-4341-b298-1d04892032f5
# ╟─120af26f-23dc-43ed-8f1c-3600c7bc2878
# ╠═fdde7445-a477-46ea-a0c7-21e7f258858c
# ╠═716832f9-1180-4fba-9e01-bda3057eb206
# ╠═8e4d6bad-32e0-4bef-a55b-a13d6f5999fe
# ╟─791912c4-fc1c-4f1a-be80-56f4d939e472
# ╠═2449fd2d-3e98-4133-aa9d-c735b4348912
# ╟─92d6333d-b605-4308-aeb2-d69e6da9ff9d
# ╠═c0756af5-4951-4d1a-9af5-25835794d944
# ╟─04586b86-a14b-44f1-bb84-e591912f7d98
# ╠═1bc8ba36-f1f0-4645-b87c-a4997452c907
# ╠═171f4942-a81e-4a7f-b0b7-61338b74e33c
# ╟─19ae5fe0-e5ee-4040-ab71-fd9ca4fd1aae
# ╟─fb271fd9-3b7f-431b-9830-5282e9792e52
# ╟─e00c7040-fc5e-4e92-922b-800ee0f13a42
# ╟─9ba3a0a3-433e-41d1-897a-c880b2da7569
# ╟─91472d8c-2e4c-4ddd-a925-9d691a854239
# ╟─d52f005e-200c-40c4-8745-7119d1774bbc
# ╟─b936727f-2d87-49b1-b8fe-4b9150fc05e1
# ╟─b7f2af2d-3c59-4fb4-a166-bdae243b7d0f
# ╟─172d2506-60b7-4da2-a433-7318425ddc08
# ╟─09930b6c-9c78-4c74-b4bb-6064745838c8
# ╟─69039968-3041-4450-855e-df23dc23218e
# ╟─1d1e0228-f3e9-4b2b-9d7e-b5839cbc929a
# ╟─412cc766-f734-4578-8169-11e185cf69c4
# ╟─4a23f7c5-2365-4852-8c83-4e523a6780a5
# ╟─6772601e-351c-4836-921e-91467b82e29b
# ╟─774660b0-a984-469d-b2b1-0998a1464df6
# ╟─d771cbaf-7025-436b-a231-5af79ebfd5b3
# ╟─48f999f0-0c0d-4fb9-babb-01bd87673530
# ╟─ad12e8cb-9bf2-4947-bb64-70179166a722
# ╟─4c5e90ec-c696-45d5-866d-c8568933a876
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
