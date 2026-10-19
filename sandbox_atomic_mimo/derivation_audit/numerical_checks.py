import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import i0e, i1e
rng=np.random.default_rng(20261014)
def cn(shape):
 return (rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)
K,P,N=3,11,5
S=cn((K,P)); G=cn((N,K)); Z=np.exp(1j*rng.uniform(-np.pi,np.pi,(N,P)))
Y=rng.normal(size=(N,P)); B=cn((N,P))
def loss(g):
 return np.sum((Y-np.abs(B)-np.real(Z*(g@S)))**2)
E=Y-np.abs(B)-np.real(Z*(G@S)); grad=-(E*np.conj(Z))@S.conj().T
Delta=cn(G.shape); h=1e-6
fd=(loss(G+h*Delta)-loss(G-h*Delta))/(2*h)
pred=2*np.real(np.vdot(grad,Delta))
result={'gradient_directional_relative_error':float(abs(fd-pred)/max(abs(fd),1))}
L=np.linalg.eigvalsh(S@S.conj().T)[-1]; eta=1/L
result['step_S9_margin']=float(loss(G)-eta*(2-L*eta)*np.linalg.norm(grad)**2-loss(G-eta*grad))
R=cn((N,P)); matrix=(R-B)@S.conj().T@np.linalg.inv(S@S.conj().T)
rows=np.zeros_like(matrix)
for n in range(N):
 rows[n,:]=np.linalg.solve(S.conj()@S.T,S.conj()@(R[n,:]-B[n,:]))
result['per_antenna_vs_matrix_LS_error']=float(np.linalg.norm(rows-matrix))
I1,I2=2,4; Na=I1*I2
T=cn((I1,I2,K)); G3=T.reshape((Na,K),order='F').T
phase=np.exp(1j*rng.uniform(-np.pi,np.pi,(P,Na)))
g=G3.reshape(-1,order='F'); A=np.diag(phase.reshape(-1,order='F'))@np.kron(np.eye(Na),S.T)
result['mode3_kronecker_error']=float(np.linalg.norm(A@g-(phase*(S.T@G3)).reshape(-1,order='F')))
sigma_r2=0.7; D=np.concatenate([A.real,-A.imag],axis=1)
FR=D.T@D/sigma_r2
full=np.trace(np.linalg.inv(FR)); block=4*sigma_r2*np.trace(np.linalg.inv(A.conj().T@A)).real
result['exact_real_CRLB_trace']=float(full)
result['block_inverse_trace']=float(block)
result['block_bound_gap']=float(full-block)
result['necessary_identifiability_counterexample']={'rank_D':int(np.linalg.matrix_rank(np.array([[1.,-1.]]))),'real_unknowns':2,'complex_block_inverse_finite':True}
# Counterexample to treating a complex amplitude multiplying cos as real.
a=np.array([1.,1j]); ph=np.array([0.,np.pi/2])
result['RWA_complex_phasor_counterexample']={'written_C_abs':float(abs(np.sum(a*np.exp(-1j*ph)))),'positive_carrier_real_phasor_C_abs':float(abs(np.sum(a.conj()*np.exp(-1j*ph))))}
# One nonzero path coefficient omitted from the dimensional equality in Eq.378.
v1=np.exp(-1j*np.arange(2)*.3);v2=np.exp(-1j*np.arange(3)*.7); beta=2+1j
result['eq378_missing_beta_error']=float(np.linalg.norm(beta*np.outer(v1,v2)-np.outer(v1,v2)))
# Rician FIM diagonal block vs full real FIM for one complex unknown.
rho=np.sqrt(5.); sig2=1.
def moment(z):
 k=2*z*rho/sig2; ratio=i1e(k)/i0e(k)
 density=2*z/sig2*np.exp(-(z-rho)**2/sig2)*i0e(k)
 return z*z*ratio*ratio*density
m=quad(moment,0,np.inf,epsabs=1e-10)[0]
betaR=(m-rho*rho)/sig2**2
v=np.array([2.,1.])/rho
result['rician_one_measurement_FIM_counterexample']={'complex_block':float(betaR),'real_FIM_rank':int(np.linalg.matrix_rank(4*betaR*np.outer(v,v))),'real_unknowns':2}
result['correct_kappa_RSR12_SNR0_K3']=float(2*10**1.2/3)
result['tex_kappa_same_case']=float(2*3*10**1.2)
assert result['gradient_directional_relative_error']<1e-6
assert result['step_S9_margin']>=-1e-9
assert result['per_antenna_vs_matrix_LS_error']<1e-10
assert result['mode3_kronecker_error']<1e-10
assert result['block_bound_gap']>0
Path(__file__).with_name('numerical_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
