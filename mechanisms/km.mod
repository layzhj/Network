TITLE Slowly activating potassium current (muscarinic K+ channel)

COMMENT
Potassium channel, Hodgkin-Huxley style kinetics.
Voltage-dependent potassium current (Im) (muscarinic K+ channel)
Slow, noninactivating.

Author: Zach Mainen, Salk Institute, 1995, zach@salk.edu
ENDCOMMENT

NEURON {
	SUFFIX km
	USEION k READ ek WRITE ik
	RANGE gbar, ik
	GLOBAL Ra, Rb
	GLOBAL q10, temp, tadj
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S) = (siemens)
	(pS) = (picosiemens)
}

PARAMETER {
	v (mV)
	celsius (degC)
	gbar = 10 (pS/cm2)
	tha  = -30 (mV)			: v1/2 for ninf
	qa   = 9 (mV)			: ninf slope
	Ra   = 0.001 (/ms)	: max act rate  (slow)
	Rb   = 0.001 (/ms)	: max deact rate  (slow)
	temp = 23	(degC)		: original temp
	q10  = 2.3				: temperature sensitivity
}

ASSIGNED {
	gk (pS/cm2)
	ek (mV)
	ik (mA/cm2)
	tadj
	ninf
	ntau (ms)
}

STATE {
	n
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gk = tadj*gbar*n
	ik = (1e-4) * gk * (v - ek)
}

DERIVATIVE states { : Computes state variable n at the current v and dt.
	rates(v)
	n' = (ninf - n)/ntau
}

INITIAL { 
	rates(v)
	n = ninf
}

FUNCTION alpha(v (mV)) (/ms) { :callable from hoc
	alpha = Ra * (v - tha) / (1 - exp(-(v - tha)/qa))
}

FUNCTION beta(v (mV)) (/ms) { :callable from hoc
	beta = -Rb * (v - tha) / (1 - exp((v - tha)/qa))
}

PROCEDURE rates(v (mV)) {
	:Computes rate and other constants at current v.
	:Call once from HOC to initialize inf at resting v.
	ntau = 1/(alpha(v) + beta(v))
	ninf = alpha(v)*ntau
	tadj = q10^((celsius - temp)/10(degC))  :temperature adjastment
}