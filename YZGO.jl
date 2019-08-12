using StaticArrays: SVector,SMatrix
using QuantumLattices

#=
n1,n2=32,32
a1,a2=[1.0,0.0],[-0.5,sqrt(3)/2]

ps=tile(reshape([0.0,0.0],2,1),[a1,a2],Tuple(Iterators.product(0:n1-1, 0:n2-1)))
points=[Point(PID(1,i),ps[:,i],[0.0,0.0]) for i=1:n1*n2]
lattice=Lattice("YZGO",points,vectors=[a1*n1,a2*n2],neighbors=1)
bond=lattice|>Bonds
config=IDFConfig{Spin}(pid->Spin(atom=1,norbital=1,spin=0.5),lattice.pids)
table=Table(config,usualspinindextotuple)
boundary=Boundary()

term=SpinTerm{2}(:J,1.0+0im,1,couplings=Heisenberg("xyz"),amplitude=bond->exp(1im*(bond|>rcoord|>azimuth)))
=#

n1,n2=64,64
a1,a2=[1.0,0.0],[0.0,1.0]
vectors=[a1,a2]
vec_al=Dict((1,1)=>[0.0,0.0])

ps=tile(reshape([0.0,0.0],2,1),[a1,a2],Tuple(Iterators.product(0:n1-1, 0:n2-1)))
points=[Point(PID(1,i),ps[:,i],[0.0,0.0]) for i=1:n1*n2]
lattice=Lattice("square",points,vectors=[a1*n1,a2*n2],neighbors=1)
bond=lattice|>Bonds
config=IDFConfig{Spin}(pid->Spin(atom=1,norbital=1,spin=0.5),lattice.pids)
table=Table(config,usualspinindextotuple)
boundary=Boundary()


sc1=SpinCoupling{2}(1.0,tags=('+','-'))
sc2=SpinCoupling{2}(1.0,tags=('z','z'))
J=SpinTerm{2}(:J,1.0+0im,1,couplings=Couplings(sc1,sc2))
B=SpinTerm{1}(:B,0.1+0im,0,couplings=Sᶻ())

disorder=(J=()->1.0+0.0*rand(),)

gen=Generator((J,B),bond,config,table,true,boundary)

var,re=StaticCMCexpand(gen)
quench!(var,0.001)
@time CSMC!(re, var,64)

F=LLGexpandF(var,re.s)
@time F(zeros(ComplexF64,size(var.state)))
f=LLG(F)
@time f(var.state)

arr=arrange(vectors,vec_al,re)
@time ClassicalMC.SDynamics(re.samples[1],f,0.01,63)
@time dre=ClassicalSpinDynamics(vectors,vec_al,re.s,arr,re.samples,f,63;dt=0.01)
@time drek=dMCresult_R2K(dre);
@time Dynamic_structure(drek,('x','x'))-Dynamic_structure(drek,'x')


