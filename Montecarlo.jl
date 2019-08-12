module ClassicalMC

"""
Classical spin Monte Carlo
"""

"""
using Printf: @printf,@sprintf
using ..Spatials: PID
using ..DegreesOfFreedom: IID,Internal,Index,FilteredAttributes,OID,Operator,LaTeX
using ..Terms: wildcard,constant,Subscript,Subscripts,subscriptexpr,Coupling,Couplings,couplingcenters,Term,TermCouplings,TermAmplitude,TermModulate
using ...Interfaces: rank,kind
using ...Prerequisites: Float,decimaltostr,delta
using ...Mathematics.VectorSpaces: VectorSpace,IsMultiIndexable,MultiIndexOrderStyle
using ...Mathematics.AlgebraOverFields: SimpleID,ID

import ..DegreesOfFreedom: script,otype,isHermitian,optdefaultlatex
import ..Terms: couplingcenter,statistics,abbr
import ...Interfaces: dims,inds,expand,matrix,permute
import ...Mathematics.AlgebraOverFields: rawelement
"""

using LinearAlgebra: cross,norm,dot
using StaticArrays
using FFTW
using QuantumLattices
using QuantumLattices.Mathematics.AlgebraOverFields
const Float=Float64


export MCvariable,staticMCresult,StaticCMCexpand
export opts2energys,deltaenergy,randCSstate,initialCSstate
export quench!, quenched!, CSMC!, CSMC

export LLG,arrange,dynamicMCresult,Rspace_dMCresult,Kspace_dMCresult
export dMCresult_R2K,dMCresult_K2R
export LLGexpandF,ClassicalSpinDynamics,Dynamic_structure

const sidtagmap=Dict(1=>'x',2=>'y',3=>'z',4=>'+',5=>'-')
const sidseqmap=Dict(v=>k for (k,v) in sidtagmap)
const sidajointmap=Dict('x'=>'x','y'=>'y','z'=>'z','+'=>'-','-'=>'+')

"""
Sopt2tuple->Tuple=(value,(tag,seq))

"""

function SIndex2tuple(seq::Int,ind::SIndex)
     return (sidseqmap[ind.tag],seq)
end
     
function Sopt2tuple(opt::SOperator,m::Dict=sidseqmap)
     val=opt.value
     inds=[]
     for oid in opt.id
          val*=oid.index.spin
          push!(inds, (m[oid.index.tag],oid.seq))
     end
     return val,inds|>Tuple
end

function Sopts2tuple(opts::Operators{T1,T2},m::Dict=sidseqmap) where {T1,T2<:SOperator}
     inds=Tuple[]
     for opt in opts|>values
          push!(inds, Sopt2tuple(opt,m))
     end
     return inds
end

#=
function genSopt2tuple(gopts::GenOperators,m::Dict=sidseqmap)
     inds=Sopts2tuple(gopts.constops,m)
     for opts in gopts.alterops
          append!(inds, Sopts2tuple(opts,m))
     end
     for opts in gopts.boundops
          append!(inds, Sopts2tuple(opts,m))
     end
     return inds
end
=#

function seqmapopts(optsd::Dict{DataType,<:Array{<:Tuple,1}})
     result=Dict{Int,Set{Int}}()
	 opts=copy(get(optsd,Number,Tuple[]))
	 append!(opts,get(optsd,Function,Tuple[]))
	 for i=1:length(opts)
		   for ind in @inbounds opts[i][2]  
				 seq=ind[2]
				 haskey(result,seq) || (result[seq]=Set{Int}())
				 push!(result[seq],i)
		   end
	 end
     Set{Int}[result[i] for i=1:length(result)]
end


#=
function Sopts2funcs(state::Array{Complex{Float},2},opts::Operators{<:ID,<:SOperator},disorder::Function,m::Dict=sidseqmap) 
     out=Function[]
     out1=Tuple[]
     for opt in opts|>values
          val,inds=Sopt2tuple(opt,m)
          push!(out1, ((val,disorder),inds))
          b=[LinearIndices(state)[ind...] for ind in inds ]
          push!(out, ()->disorder()*val*prod(state[b]))
     end
     return out,Dict(Function=>out1)
end
=#

function Sopts2funcs(state::Array{Complex{Float},2},opts::Operators{<:ID,<:SOperator},disorder::Function,m::Dict=sidseqmap) 
     out=Function[]
     out1=Tuple[]
     for opt in opts|>values
          val,inds=Sopt2tuple(opt,m)
          push!(out1, (:($disorder()*$val),inds))
          b=[LinearIndices(state)[ind...] for ind in inds ]
          push!(out, ()->disorder()*val*prod(state[b]))
     end
     return out,Dict(Function=>out1)
end


function Sopts2funcs(state::Array{Complex{Float},2},opts::Operators{<:ID,<:SOperator},m::Dict=sidseqmap) 
     out=Function[]
     out1=Tuple[]
     for opt in opts|>values
          val,inds=Sopt2tuple(opt,m)
          push!(out1, (val,inds))
          b=[LinearIndices(state)[ind...] for ind in inds ]
		  push!(out,  ()->val*prod(state[b]))
     end
     return out,Dict(Number=>out1)
end

function optsTuple2funcs(state::Array{Complex{Float},2},optsd::Dict{DataType,<:Array{<:Tuple,1}})
     out=Function[]
	 opts=copy(get(optsd,Number,Tuple[]))
	 append!(opts,get(optsd,Function,Tuple[]))
	 for (val,inds) in opts
		  b=[LinearIndices(state)[ind...] for ind in inds ]
		  typeof(val)<:Number ?  push!(out, ()->val*prod(state[b]))  :  
					             push!(out, ()->val[2]()*val[1]*prod(state[b]))
	 end
     return out
end

"""
AbstractGenerator->Function[opt],Tuple[opt],seqmapopts(opt{Tuple})
"""
function gen2optfunc(state::Array{Complex{Float},2},gen::AbstractGenerator,disorder::Union{Nothing,Function,NamedTuple}=nothing)
      if disorder==nothing
            opts=expand(gen)
            out,outc=Sopts2funcs(state,opts)
            return out,outc,seqmapopts(outc)
      elseif disorder|>typeof<:NamedTuple
            out=Function[]
            names=disorder|>keys
            unames=setdiff(gen.terms|>keys,names)
            for name in unames
                  opts=expand(gen,name)
                  outT,outc=Sopts2funcs(state,opts) 
                  append!(out,outT)
            end
            for name in names
                  opts=expand(gen,name)
                  outT,outcT=Sopts2funcs(state,opts,disorder[name]) 
                  append!(out,outT)
                  merge!(outc, outcT)
            end
            return out,outc,seqmapopts(outc)
      else
            opts=expand(gen)
            out,outc=Sopts2funcs(state,opts,disorder)
            return out,outc,seqmapopts(outc)
      end
end


"""
initalstate
"""
function randCSstate()
    z=1.0-rand()*2.0
	theta=rand()*2.0*pi
	r=sqrt(1.0-z^2)
	x=cos(theta)*r
	y=sin(theta)*r
	Complex{Float}[x,y,z,x+y*1.0im,x-y*1.0im]
end

function initialCSstate(n::Int, static::Union{Nothing,Array{Complex{Float},1},Function}=nothing)
      if static==nothing
         out=zeros(Complex{Float},5,n)
         for i=1:n
              out[:,i]=randCSstate()
         end
         return out
      elseif static|>typeof<:Function
         out=zeros(Complex{Float},5,n)
         for i=1:n
              out[:,i]=static(i)
         end
         return out
      else
            return repeat(static, 1, n)
      end
end

function opts2energys(opts::Array{<:Tuple,1},state::Array{Complex{Float},2})
            energys=Float[]
            for opt in opts
                 energy=opt[1]
                 for ind in opt[2]
                       @inbounds energy*=state[ind...]
                 end
                 push!(energys,real(energy))
            end
            return energys
end

function deltaenergy(opts::Array{<:Tuple,1},state::Array{Complex{Float},2},maps::Set{Int}...)
            map=union(maps...)|>collect|>sort
            energys=Float[]
            for opt in map
                 @inbounds energy=opts[opt][1]
                 for ind in @inbounds opts[opt][2]
                       @inbounds energy*=state[ind...]
                 end
                 push!(energys,real(energy))
            end
            return map,energys
end


function deltaenergy(opts::Array{<:Function,1},maps::Set{Int}...)
            maps=union(maps...)|>collect|>sort
            return maps,real(map.(opts[maps]))
end

#=
function deltaenergy(opts::Array{<:Function,1},maps::Set{Int}...)
            map=union(maps...)|>collect|>sort
			energys=Float[]
			for i in map
				push!(energys,real(opts[i]()))
			end
            #energys=[real(opts[i]()) for i in map]
            return map,energys
end
=#
function space(rtable::Dict{Int,<:Set{<:Index}},points::Array{<:Point,1},idf::IDFConfig)
      m=rtable|>length
      atom_orbitl=[]
	  s=Float[]
      r=[]
      for i=1:m
           ind=pop!(rtable[i])
		   push!(s,ind.spin)
           push!(atom_orbitl,(idf[ind|>pid].atom,ind.orbital))
           for pt in points
                if ind|>pid==pt.pid
                   push!(r,pt.rcoord)
                   break
                end
           end
      end
      @assert m==length(r) "the length of spins is erreor"
      return s, atom_orbitl|>Tuple, r|>Tuple
end

struct staticMCresult{N,O<:Array{<:AbstractArray,1},P<:Tuple,Q<:Tuple}
      samples::O
      energys::Array{Float,1}
	  s::Array{Float,1}
      r::P
      atom_orbitl::Q
      function staticMCresult(samples,energys::Array{Float,1},s::Array{Float,1},r::NTuple{N,Any},s_l::NTuple{N,Any}) where {N}
            new{N,typeof(samples),typeof(r),typeof(s_l)}(samples,energys,s,r,s_l)
      end
end   

Base.length(res::staticMCresult)=(res.samples)|>length
rank(res::staticMCresult)=rank(res|>typeof)
rank(::Type{<:staticMCresult{N}}) where N=N

function Base.append!(res1::staticMCresult,res2::staticMCresult)
	@assert res1.r==res2.r "the spaces do not match"
    append!(res1.samples,res2.samples)
    append!(res1.energys,res2.energys)
    return res1
end

struct  MCvariable{O<:Array{<:Function,1},OT<:Dict{DataType,<:Array{<:Tuple,1}},D<:Union{Nothing,Function,NamedTuple}}
      state::Array{Complex{Float64},2}
      optenergys::Array{Float,1}
      seqmapopt::Array{Set{Int},1}
      opts::O
      opt_Tuples::OT
      half::Bool
      disorder::D
      T::Array{Float,1}
      function MCvariable(state,optenergys,seqmapopt,opts::Array{<:Function,1},optsT::Dict{DataType,<:Array{<:Tuple,1}},half,disorder::Union{Nothing,Function,NamedTuple},T=Float[1000.0])
            new{typeof(opts),typeof(optsT),typeof(disorder)}(state,optenergys,seqmapopt,opts,optsT,half,disorder,T)
      end
end

function MCvariable(gen::AbstractGenerator,disorder::Union{Nothing,Function,NamedTuple}=nothing)
      rtable=gen.table|>reverse
      state= initialCSstate(rtable|>length)
      opts,optsT,seqmapopt=gen2optfunc(state,gen,disorder)
      MCvariable(state,real(map.(opts)),seqmapopt,opts,optsT,gen.half,disorder)
end


function refresh!(var::MCvariable)
      var.opts[:]=optsTuple2funcs(var.state,var.opt_Tuples)
end

function Base.push!(res::staticMCresult,values::MCvariable...)
     for x in values
          #sample=SMatrix{size(x.state)...}(x.state)
          push!(res.samples,copy(x.state))
          push!(res.energys,(x.half ? 2.0*sum(x.optenergys) : sum(x.optenergys)))
     end
     return res
end

function StaticCMCexpand(gen::AbstractGenerator,disorder::Union{Nothing,Function,NamedTuple}=nothing)
      rtable=gen.table|>reverse
      s,s_l,r=space(rtable,gen.bonds.bonds[1],gen.config)
      state= initialCSstate(rtable|>length)
      opts,optsT,seqmapopt=gen2optfunc(state,gen,disorder)
      out1=MCvariable(state,real(map.(opts)),seqmapopt,opts,optsT,gen.half,disorder)
      out2=staticMCresult((state|>typeof)[],Float[],s,r,s_l)
      return out1,out2
end




"""
Monte Carlo procedure
"""


function point_rand_update!(var::MCvariable,T::Float=var.T[end])
     state=var.state
     energys=var.optenergys
     smt=var.seqmapopt
     opts=var.opts

     p=rand(1:length(smt))
     @inbounds temp=state[:,p]
     @inbounds state[:,p]=randCSstate()
     seq,tempE=deltaenergy(opts,smt[p])
     @inbounds de=sum(energys[seq].-tempE)
     if exp(de/T)>rand()
          @inbounds energys[seq]=tempE
          return de
     else
          @inbounds state[:,p]=temp
          return 0.0
     end
end

function point_rand_update(var::MCvariable,T::Float=var.T[end])
     var1=deepcopy(var)
     state=var1.state
     energys=var1.optenergys
     smt=var1.seqmapopt
     opts=var1.opts

     p=rand(1:length(smt))
     temp=state[:,p]
     state[:,p]=randCSstate()
     seq,tempE=deltaenergy(opts,smt[p])
     de=sum(energys[seq].-tempE)
     if exp(de/T)>rand()
          energys[seq]=tempE
          return var1
     else
          state[:,p]=temp
          return var1
     end
end

function quench!(var::MCvariable,T::Float=0.01,pace::Int=length(var.seqmapopt);renew::Function=point_rand_update!)
     if var.T[end]<=T
          push!(var.T,T)
          return true
     end
     for t in LinRange(min(pace*T,var.T[end]), T, 50*pace)
          renew(var,t)
     end
     quenched!(var,T,pace,renew=renew)
end

function quenched!(var::MCvariable,T::Float=0.01,pace::Int=length(var.seqmapopt);renew::Function=point_rand_update!,mn::Int=10)
     if mn<1
           push!(var.T,T)
           @warn "The process of quench was shut down at $T"
           return true
     end
     x=(-pace/2.0+0.5):(pace/2.0-0.5)
     y=cumsum([renew(var,T) for _=1:pace])
     k=sum(x.*(y.-sum(y)/pace))/sum(x.^2)
     if abs(k)<T
           push!(var.T,T)
           return true
     else
           quenched!(var,T,pace,renew=renew,mn=mn-1)
     end
end


function CSMC!(result::staticMCresult, var::MCvariable, m::Int=64;pace::Int=result.r|>length,renew::Function=point_rand_update!)
     for _=1:m
          for _=1:pace
               renew(var)
          end
          push!(result,var)
     end
     return result
end

function CSMC(var::MCvariable,s::Array{Float,1},r::Tuple,s_l::Tuple,m::Int=100;pace::Int=r|>length,renew::Function=point_rand_update!)
     out=staticMCresult((var.state|>typeof)[],Float[],s,r,s_l)
     for _=1:m
          for _=1:pace
               renew(var)
          end
          push!(out,var)
     end
     return out
end

"""
 Landau–Lifshitz–Gilbert  spin dynamic
"""

function D_optsT(var::MCvariable,s::Vector{Float64})
	opts=copy(get(var.opt_Tuples,Number,Tuple[]))
	append!(opts,get(var.opt_Tuples,Function,Tuple[]))
	n=var.seqmapopt|>length
    result=Dict((i,j)=>Tuple[] for i=1:5,j=1:n)
	for opt in opts
		for (i,ind) in pairs(opt[2])
			val=opt[1]
			typeof(val)==Expr ? val.args[end]/=s[ind[2]] : val/=s[ind[2]]
			push!(result[ind],(val,deleteat!(opt[2]|>collect,i)|>Tuple))
		end
	end
	for i=1:n
		for opt in result[(4,i)]
			val=opt[1]
			push!(result[(1,i)],(val,opt[2]))
			typeof(val)==Expr ? val.args[end]*=1.0im : val*=1.0im
			push!(result[(2,i)],(val,opt[2]))
		end
		for opt in result[(5,i)]
			val=opt[1]
			push!(result[(1,i)],(val,opt[2]))
			typeof(val)==Expr ? val.args[end]*=-1.0im : val*=-1.0im
			push!(result[(2,i)],(val,opt[2]))
		end
	end
	[[result[(i,j)] for i=1:3] for j=1:n]
end



function innerF(o)
	out=Expr(:call,:+)
	for p in o
	    out1=Expr(:call,:*)
		push!(out1.args,p[1])
		for pp in p[2]
			push!(out1.args,:(A[$pp...]))
		end
		push!(out.args,out1)
	end
	out	
end

function LLGexpandF(var::MCvariable,s::Vector{Float64})
	F_Tuples=D_optsT(var,s)
	ex=Expr(:ref,Vector{Float})
	for opt in F_Tuples
		ex1=Expr(:ref,Float)
		for o in opt
			var.half ? push!(ex1.args,Expr(:call,:*,2.0,Expr(:call,:real,innerF(o)))) : push!(ex1.args,Expr(:call,:real,innerF(o)))
		end
		push!(ex.args,ex1)
	end
	return @eval A->$ex
end


#=
function D_optsT(var::MCvariable,s::Vector{Float64})
	opts=var.opt_Tuples
	n=var.seqmapopt|>length
	if haskey(opts,Function)
		result=Dict((i,j)=>(Tuple[],Tuple[]) for i=1:5,j=1:n)
		for opt in opts[Number]
			val=opt[1]
			for (i,ind) in pairs(opt[2])
				push!(result[ind][1],(val/s[ind[2]],deleteat!(opt[2]|>collect,i)|>Tuple))
			end
		end
		for opt in opts[Function]
			val=opt[1][1]
			f=opt[1][2]
			for (i,ind) in pairs(opt[2])
				push!(result[ind][2],((val/s[ind[2]],f),deleteat!(opt[2]|>collect,i)|>Tuple)) 
			end
		end
		
		for i=1:n
			for opt in result[(4,i)][1]
				push!(result[(1,i)][1],opt)
				push!(result[(2,i)][1],(1.0im*opt[1],opt[2]))
			end
			for opt in result[(5,i)][1]
				push!(result[(1,i)][1],opt)
				push!(result[(2,i)][2],(-1.0im*opt[1],opt[2]))
			end	
			for opt in result[(4,i)][2]
				push!(result[(1,i)][2],opt)
				push!(result[(2,i)][2],((1.0im*opt[1][1],opt[1][2]),opt[2]))
			end
			for opt in result[(5,i)][2]
				push!(result[(1,i)][2],opt)
				push!(result[(2,i)][2],((-1.0im*opt[1][1],opt[1][2]),opt[2]))
			end
		end
		return [[result[(1,j)],result[(2,j)],result[(3,j)]] for j=1:n]
	else
		result=Dict((i,j)=>Tuple[] for i=1:5,j=1:n)
		for opt in opts[Number]
			val=opt[1]
			for (i,ind) in pairs(opt[2])
				push!(result[ind],(val/s[ind[2]],deleteat!(opt[2]|>collect,i)|>Tuple))
			end
		end
		for i=1:n
			for opt in result[(4,i)]
				push!(result[(1,i)],opt)
				push!(result[(2,i)],(1.0im*opt[1],opt[2]))
			end
			for opt in result[(5,i)]
				push!(result[(1,i)],opt)
				push!(result[(2,i)],(-1.0im*opt[1],opt[2]))
			end
		end
		return [[result[(1,j)],result[(2,j)],result[(3,j)]] for j=1:n]
	end
end

function fnumber(S::Array{Complex{Float},2},fxyzs::Array{Tuple,1})
	out=0.0
	for fxyz in fxyzs
		out1=fxyz[1]
		for ind in fxyz[2]
			out1*=S[ind...]
		end
		out+=out1
	end
	return real(out)
end

function ffunction(S::Array{Complex{Float},2},fxyzs::Array{Tuple,1})
	out=0.0
	for fxyz in fxyzs
		out1=fxyz[1][1]*fxyz[1][2]()
		for ind in fxyz[2]
			out1*=S[ind...]
		end
		out+=out1
	end
	return real(out)
end

function LLGexpandF(var::MCvariable,s::Vector{Float64})
	F_Tuples=D_optsT(var,s)
	opts=var.opt_Tuples
    if var.half
		if haskey(opts,Function)
			function ff1(S::Array{Complex{Float},2})
				out=Vector{Float}[]
				for fxyz in F_Tuples
					push!(out,Float[2.0*(fnumber(S,fxyz[1][1])+ffunction(S,fxyz[1][2])),2.0*(fnumber(S,fxyz[2][1])+ffunction(S,fxyz[2][2])),2.0*(fnumber(S,fxyz[3][1])+ffunction(S,fxyz[3][2]))])
				end
				return out
			end
			return ff1
		else
			function ff2(S::Array{Complex{Float},2})
				out=Vector{Float}[]
				for fxyz in F_Tuples
					push!(out,Float[2.0*fnumber(S,fxyz[1]),2.0*fnumber(S,fxyz[2]),2.0*fnumber(S,fxyz[3])])
				end
				return out
			end
			return ff2
		end
	else
		if haskey(opts,Function)
			function ff3(S::Array{Complex{Float},2})
				out=Vector{Float}[]
				for fxyz in F_Tuples
					push!(out,Float[(fnumber(S,fxyz[1][1])+ffunction(S,fxyz[1][2])),(fnumber(S,fxyz[2][1])+ffunction(S,fxyz[2][2])),(fnumber(S,fxyz[3][1])+ffunction(S,fxyz[3][2]))])
				end
				return out
			end
			return ff3
		else
			function ff4(S::Array{Complex{Float},2})
				out=Vector{Float}[]
				for fxyz in F_Tuples
					push!(out,Float[fnumber(S,fxyz[1]),fnumber(S,fxyz[2]),fnumber(S,fxyz[3])])
				end
				return out
			end
			return ff4
		end
	end
end
=#



function LLG(F::Function)
    function (S::Array{Complex{Float},2})
		out=Array{Complex{Float}}(undef,size(S))
		Fm=F(S)
		for i=1:length(Fm)
			@inbounds out[1:3,i]=cross(real(S[1:3,i]),Fm[i])
		end
		@inbounds out[4,:]=out[1,:]+1.0im*out[2,:]
		@inbounds out[5,:]=out[1,:]-1.0im*out[2,:]
		return out
	end
end

function LLG(F::Function,al::Float)
    function (S::Array{Complex{Float},2})
		out=Array{Complex{Float}}(undef,size(S))
		Fm=F(S)
		for i=1:length(Fm)
			@inbounds sxyz=real(S[1:3,i])
			dsxyz=cross(sxyz,Fm[i])
			@inbounds out[1:3,i]=dsxyz+al*cross(dsxyz,sxyz)
		end
		@inbounds out[4,:]=out[1,:]+1.0im*out[2,:]
		@inbounds out[5,:]=out[1,:]-1.0im*out[2,:]
		return out
	end
end


#=
function LLG(F::Function)
    function (S::Array{Complex{Float},2})
		sxyz=[real(S[1:3,i]) for i=1:size(S)[2]]
		ds=hcat(cross.(sxyz,F(S))...)
		vcat(ds,ds[1:1,:]+1.0im*ds[2:2,:],ds[1:1,:]-1.0im*ds[2:2,:])
	end
end

function LLG(F::Function,al::Float)
    function (S::Array{Complex{Float},2})
		sxyz=[real(S[1:3,i]) for i=1:size(S)[2]]
		dsxyz=cross.(sxyz,F(S))
		dsxyz=dsxyz+al*cross.(dsxyz,sxyz)
		ds=hcat(dsxyz...)
		vcat(ds,ds[1:1,:]+1.0im*ds[2:2,:],ds[1:1,:]-1.0im*ds[2:2,:])
	end
end
=#

function amendS!(S::Array{Complex{Float},2})
	for i=1:size(S)[2]
		@inbounds S[:,i]=S[:,i]/norm(S[1:3,i])
	end
end

function RungeKutta4(y::Array{Complex{Float},2},dt::Float,F::Function)
	k1=F(y)
	k2=F(y+dt*k1/2.0)
	k3=F(y+dt*k2/2.0)
	k4=F(y+dt*k3)
	y+dt*(k1+2.0*k2+2.0*k3+k4)/6.0		
end


#=
function RungeKutta4(y::Array{Complex{Float},2},t::Float,dt::Float,F::Function)
	k1=F(y,t)
	k2=F(y+k1*dt/2.0,t+dt/2.0)
	k3=F(y+k2*dt/2.0,t+dt/2.0)
	k4=F(y+dt*k3,t+dt)
	y+dt*(k1+2.0*k2+2.0*k3+k4)/6.0		
end
=#



function SDynamics(s::Array{Complex{Float},2},llg::Function,dt::Float,T::Int,numerical::Function=RungeKutta4)
	s1=s
	out=Array{Complex{Float}}(undef,size(s)...,T+1)
	out[:,:,1]=s1
	for i=2:T+1
		s1=numerical(s1,dt,llg)
		amendS!(s1)
		out[:,:,i]=s1
	end
	out
end

#=
function SDynamics(s::Array{Complex{Float},2},llg::Function,dt::Float,T::Int,numerical::Function=RungeKutta4)
	s1=copy(s)
	for _=1:T
		s1=numerical(s1,dt,llg)
		amendS!(s1)
		s=cat(s,s1,dims=3)
	end
	s
end

function SDynamics(s::Array{Complex{Float},2},llg::Function,T::AbstractRange,numerical::Function=RungeKutta4)
	dt=T.step|>Float
	s1=copy(s)
	for t in T
		s1=numerical(s1,t,dt,llg)
		amendS!(s1)
		s=cat(s,s1,dims=3)
	end
	s
end

function SDynamics(s::Array{Complex{Float},2},llg::Function,dt::Array{<:Real,1},T0::Float,numerical::Function=RungeKutta4)
	T=T0-dt[1]+cumsum(dt)
	s1=copy(s)
	for (t,d) in zip(T,dt)
		s1=numerical(s1,t,d,llg)
		amendS!(s1)
		s=cat(s,s1,dims=3)
	end
	s
end
=#




struct dynamicMCresult{kind,L,N,O<:Dict{<:Tuple,<:Array{<:AbstractArray}},Q<:Dict}
      samples::O
      vectors::Vector{SVector{N,Float}}
	  #reciprocals::Vector{SVector{N,Float}}
      dvec::Q
	  del::Float
      function dynamicMCresult{k,L}(samples,vectors::AbstractVector{<:AbstractVector{<:Real}},vec_al,dt) where {k,L}
			N=length(vectors)
			vectors=convert(Vector{SVector{N,Float}},vectors)
			#recipls=convert(Vector{SVector{N,Float}},reciprocals(vectors))
            new{k,L,N,typeof(samples),typeof(vec_al)}(samples,vectors,vec_al,dt)
      end
end   

Base.length(res::dynamicMCresult)=Base.length(res|>typeof)
Base.length(::Type{<:dynamicMCresult{k,L}}) where {k,L}=L
Base.size(res::dynamicMCresult)=(res.samples|>values|>collect)[1][1]|>size
dims(res::dynamicMCresult)=dims(res|>typeof)
dims(::Type{<:dynamicMCresult{k,L,N}}) where {k,L,N}=N
kind(res::dynamicMCresult)=kind(res|>typeof)
kind(::Type{<:dynamicMCresult{k}}) where k=k

const Rspace_dMCresult{L,N,O<:Dict,Q<:Dict}=dynamicMCresult{:R,L,N,O,Q}
const Kspace_dMCresult{L,N,O<:Dict,Q<:Dict}=dynamicMCresult{:K,L,N,O,Q}


function arrange(vectors::AbstractVector{<:AbstractVector{<:Real}},vec_al,re::staticMCresult) 
	N=length(vectors)
	a_l=re.atom_orbitl
	r=re.r
	out=nothing
	for (k,d) in vec_al
		lis=findall(x->x==k,a_l)
		shape=Tuple{Vararg{Int,N}}[]
		for i in lis
			push!(shape,round.(Int,decompose(r[i]-d,vectors...)).+1)
		end
		nn=Int[maximum([sh[i] for sh in shape]) for i=1:N]
		@assert prod(nn)==length(shape) "the dimensions do not match!"
		shape=Int[sh[1]+sum((sh[2:end].-1).*cumprod(nn)[1:end-1]) for sh in shape]
		if out==nothing
			out=Dict(k=>(lis,shape,nn))
		else
			out[k]=(lis,shape,nn)
		end
	end
	function (samples::AbstractVector{<:Array})
		dim=size(samples[1])
		#Dict(k=>[SArray{Tuple{var[3]...,dim[end-1],dim[end]}}((tem=sam[var[1],:,:];tem[var[2],:,:]=tem)) for sam in samples] for (k,var) in out)
		Dict(k=>[reshape((tem=sam[var[1],:,:];tem[var[2],:,:]=tem),var[3]...,dim[end-1],dim[end]) for sam in samples] for (k,var) in out)
	end
end



function dMCresult_R2K(res::Rspace_dMCresult)
	out=copy(res.samples)
	N=dims(res)
	#sz=size(res)
	for (k,sam) in out
		#out[k]=[fft(s,1:N+1) for s in sam]
		out[k]=Array{Complex{Float},N+2}[(t=fft(selectdim(s,ndims(s),1:3),1:N+1);cat(t,selectdim(t,ndims(t),1)+1.0im*selectdim(t,ndims(t),2),selectdim(t,ndims(t),1)-1.0im*selectdim(t,ndims(t),2),dims=ndims(t))) for s in sam]
	end
	recipls=convert(Vector{SVector{N,Float}},reciprocals(res.vectors))
	dw=2*pi/res.del/size(res)[N+1]
	dynamicMCresult{:K,length(res)}(out,recipls,res.dvec,dw)
end

function dMCresult_K2R(res::Kspace_dMCresult)
	out=copy(res.samples)
	N=dims(res)
	#sz=size(res)
	for (k,sam) in out
		#out[k]=[ifft(s,1:N+1) for s in sam]
		out[k]=Array{Complex{Float},N+2}[(t=ifft(selectdim(s,ndims(s),1:3),1:N+1);cat(t,selectdim(t,ndims(t),1)+1.0im*selectdim(t,ndims(t),2),selectdim(t,ndims(t),1)-1.0im*selectdim(t,ndims(t),2),dims=ndims(t))) for s in sam]
	end
	recipls=convert(Vector{SVector{N,Float}},reciprocals(res.vectors))
	dt=2*pi/res.del/size(res)[N+1]
	dynamicMCresult{:R,length(res)}(out,recipls,res.dvec,dt)
end

function Base.append!(res1::dynamicMCresult,res2::dynamicMCresult)
	@assert kind(res1)==kind(res1) "different kinds"
    @assert res1.vectors==res2.vectors && res1.dvec==res2.dvec "the spaces do not match"
	for k in res1.samples|>keys
		append!(res1.samples[k],res2.samples[k])
	end
    return res1
end

function ClassicalSpinDynamics(vectors::AbstractVector{<:AbstractVector{<:Real}},vec_al::Dict,spin::Array{Float},arr::Function,re::Vector{<:Array{Complex{Float}}},F::Function,T::Int=64;dt::Float,numerical::Function=RungeKutta4)
	#f=LLG(F)
	y=Array{Complex{Float},3}[]
	for s in re
		push!(y,spin.*permutedims(SDynamics(s,F,dt,T,numerical), [2,3,1]))
	end
	#y=[spin.*permutedims(SDynamics(s,F,dt,T,numerical), [2,3,1]) for s in re]
	#y=[re.s.*permutedims(sam, [2,3,1]) for sam in y]
	#arr=arrange(vectors,vec_al,re) 
	dynamicMCresult{:R,length(y)}(arr(y),vectors,vec_al,dt)
	#Rspace_dMCresult(y,vectors,vec_al,re.s,re.r,re.atom_orbitl,dt)
end




#=

function ClassicalSpinDynamics(vectors::AbstractVector{<:AbstractVector{<:Real}},vec_al::Dict,re::staticMCresult,F::Function,T::Int=64;dt::Float,numerical::Function=RungeKutta4)
	f=LLG(F)
	y=[SDynamics(s,f,dt,T,numerical) for s in re.samples]
	y=[re.s.*permutedims(sam, [2,3,1]) for sam in y]
	arr=arrange(vectors,vec_al,re) 
	dynamicMCresult{:R,length(y)}(arr(y),vectors,vec_al,dt)
	#Rspace_dMCresult(y,vectors,vec_al,re.s,re.r,re.atom_orbitl,dt)
end

function ClassicalSpinDynamics(vectors::AbstractVector{<:AbstractVector{<:Real}},vec_al::Dict,re::staticMCresult,F::Function,T::AbstractRange;numerical::Function=RungeKutta4)
	f=LLG(F)
	y=[SDynamics(Array(s),f,T,numerical) for s in re.samples]
	y=[re.s.*permutedims(sam, [2,3,1]) for sam in y]
	arr=arrange(vectors,vec_al,re)
	dynamicMCresult{:R,length(y)}(arr(y),vectors,vec_al,dt)	
	#Rspace_dMCresult(y,vectors,vec_al,re.s,re.r,re.atom_orbitl,T.step|>Float)
end



function ClassicalSpinDynamics(vectors::AbstractVector{<:AbstractVector{<:Real}},vec_al::Dict,re::staticMCresult,F::Function,dt::Array{<:Real,1},T0::Float=0.0;numerical::Function=RungeKutta4)
	f=LLG(F)
	y=[SDynamics(Array(s),f,dt,T0,numerical) for s in re.samples]
end

=#

"""
 correlations of classical spins
"""

function structure_factor(dr,k,N::Int...)
	pace=StepRangeLen.(0,k./N,N)
	[exp(1.0im*dot(sum(i),dr)) for i=Iterators.product(pace...)] 
end


function Dynamic_structure(drek::Kspace_dMCresult,inds::Tuple{Char,Char}...)
	inds=[(sidseqmap[sidajointmap[ind[1]]],sidseqmap[ind[2]]) for ind in inds]
	N=dims(drek)
	len=length(drek)
	sz=size(drek)[1:end-1]
	psz=prod(sz)
	samples=drek.samples
	dvec=copy(drek.dvec)
	origin=(dvec|>keytype)[]
	for (k,dr) in dvec
		isapprox(norm(dr),0.0) && (pop!(dvec, k);push!(origin,k))
	end
	dk=Dict(k=>structure_factor(dr,drek.vectors,sz[1:N]...) for (k,dr) in dvec)
	Sout=zeros(Complex{Float},sz)
	for i=1:len
		S=zeros(Complex{Float},sz)
		for ind in inds
			Ad=zeros(Complex{Float},sz)
			A=zeros(Complex{Float},sz)
			for k in origin
				sam=samples[k][i]
				Ad+=selectdim(sam,ndims(sam),ind[1])
				A+=selectdim(sam,ndims(sam),ind[2])
			end
			for (k,ek) in dk
				sam=samples[k][i]
				Ad+=ek.*selectdim(sam,ndims(sam),ind[1])
				A+=ek.*selectdim(sam,ndims(sam),ind[2])
			end
			S+=(conj(Ad).*A)/psz^2
		end
		Sout+=S
	end
	Sout/len
end

function Dynamic_structure(drek::Kspace_dMCresult,inds::Char...)
	inds=[sidseqmap[ind] for ind in inds]
	N=dims(drek)
	len=length(drek)
	sz=size(drek)[1:end-1]
	psz=prod(sz)
	samples=drek.samples
	dvec=copy(drek.dvec)
	origin=(dvec|>keytype)[]
	for (k,dr) in dvec
		isapprox(norm(dr),0.0) && (pop!(dvec, k);push!(origin,k))
	end
	dk=Dict(k=>structure_factor(dr,drek.vectors,sz[1:N]...) for (k,dr) in dvec)
	Sout=zeros(sz)
	for i=1:len
		S=zeros(sz)
		for ind in inds
			A=zeros(Complex{Float},sz)
			for k in origin
				sam=samples[k][i]
				A+=selectdim(sam,ndims(sam),ind)
			end
			for (k,ek) in dk
				sam=samples[k][i]
				A+=ek.*selectdim(sam,ndims(sam),ind)
			end
			S+=(abs.(A).^2)/psz^2
		end
		Sout+=S
	end
	Sout/len
end


end