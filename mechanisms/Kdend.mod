COMMENT
Potassium current for the dendrites
ENDCOMMENT

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
}
 
NEURON {
    SUFFIX Kdend
    USEION k READ ek WRITE ik
    RANGE gkdend, ik
	GLOBAL ninf, ntau
}
 
PARAMETER {
    v (mV)
	celsius = 24 (degC)
    gkdend = .0230 (mho/cm2)
    ek = -90 (mV)
}
 
STATE { n }
 
ASSIGNED {
	ik (mA/cm2)
	ninf 
	ntau (ms)
}
 
INITIAL {	
	n = ninf 
}

BREAKPOINT {
    SOLVE states METHOD cnexp
	ik = gkdend*pow(n, 4)*(v - ek)    
}

DERIVATIVE states {	:exact when v held constant
	rates(v)
	n' = (ninf - n)/ntau 
}
UNITSOFF

PROCEDURE rates(v(mV)) {  :Computes rate and other constants at current v.
    
    :Call once from HOC to initialize inf at resting v.
    LOCAL alpha, beta
	
	alpha = 0.018*vtrap(-(v-20),21)
	beta  = 0.0036*vtrap(v-30,12)
	ntau  = 1/(alpha + beta)
	ninf  = alpha*ntau
}

FUNCTION vtrap(x (mV), y (mV)) (1) {
	:Traps for 0 in denominator of rate eqns.
	if (fabs(x/y) < 1e-6) {
		vtrap = 1(/mV)*y*(1 - x/y/2)
	} else {
		vtrap = 1(/mV)*x/(Exp(x/y) - 1)
	}
}

FUNCTION Exp(x) {
	if (x < -100) {
		Exp = 0
	}else{
		Exp = exp(x)
	}
}
UNITSON
