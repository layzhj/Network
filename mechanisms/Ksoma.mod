COMMENT
Potassium current for the soma
ENDCOMMENT

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
}
 
NEURON {
    SUFFIX Ksoma
    USEION k READ ek WRITE ik
    RANGE gksoma, ik
	GLOBAL ninf, ntau
}
 
PARAMETER {
	v (mV)
	celsius = 24 (degC)
    gksoma = .0319 (mho/cm2)
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
	ik = gksoma*n*n*n*n*(v - ek)    
}

DERIVATIVE states {	:exact when v held constant
	rates(v)
	n' = (ninf - n) / ntau
}
UNITSOFF

PROCEDURE rates(v(mV)) {  
	:Computes rate and other constants at current v.
    
    :Call once from HOC to initialize inf at resting v.
    LOCAL alpha, beta
	
	alpha = 0.018*vtrap(-(v-25),25)
	beta  = 0.0036*vtrap(v-35,12)
	ntau  = 1/(alpha + beta)
	ninf  = alpha*ntau
}

FUNCTION vtrap(x,y) {	:Traps for 0 in denominator of rate eqns.
	if (fabs(x/y) < 1e-6) {
		vtrap = y*(1 - x/y/2)
	} else {
		vtrap = x/(Exp(x/y) - 1)
	}
}

FUNCTION Exp(x) {
	if (x < -100) {
		Exp = 0
	} else {
		Exp = exp(x)
	}
}
UNITSON
