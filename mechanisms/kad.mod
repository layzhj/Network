TITLE A-type K+ channel

COMMENT
Original from Klee, et al., 1995, J Neurophysiol, 74, 1892-1895, doi: 10.1152/jn.1995.74.5.1982
modified to account for Dax A Current, Migliore et al., 1997, J. Comput. Neurosci. 7, 5-15, doi: 10.1023/a:1008906225285
modified by Poirazi on 10/2/00 according to Hoffman et al., 1997, Nature 387, 869-875, doi: 10.1038/43119
to account for I_A distal (>100microns)
(n) activation, (l) inactivation
ENDCOMMENT

NEURON {
	SUFFIX kad
	USEION k READ ek WRITE ik
	RANGE gkabar, gka
	GLOBAL ninf,linf,taul,taun
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	v (mV)
	ek (mV)
	gkabar = 0.018 (mho/cm2)
}

ASSIGNED {
	gka (mho/cm2)
	ik (mA/cm2)
	ninf
	linf
	taun (ms)
	taul (ms)
}

STATE {
	n
	l
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gka = gkabar*pow(n, 4)*l
	ik = gka*(v-ek)
}

DERIVATIVE states {
	rates(v)
	n' = (ninf - n)/taun
	l' = (linf - l)/taul
}

INITIAL {
	rates(v)
	n = ninf
	l = linf
	gka = gkabar*pow(n, 4)*l
	ik = gka*(v-ek)
}

FUNCTION alpn(v (mV)) (/ms) {
	alpn = -0.01*(v+34.4)/(exp((v+34.4)/-21)-1)
}

FUNCTION betn(v (mV)) (/ms) {
	betn = 0.01*(v+34.4)/(exp((v+34.4)/21)-1)
}

FUNCTION alpl(v (mV)) (/ms) {
	alpl = -0.01*(v+58)/(exp((v+58)/8.2)-1)
}

FUNCTION betl(v (mV)) (/ms) {
	betl = 0.01*(v+58)/(exp((v+58)/-8.2)-1)
}

PROCEDURE rates(v (mV)) {
	:callable from hoc
	ninf = alpn(v)/(alpn(v) + betn(v))
	taun = 0.2

	linf = alpl(v)/(alpl(v) + betl(v))
	if (v > -20) {
		taul = 5(ms) + 2.6(ms)*(v+20(mV))/10(mV)
	} else {
		taul = 5
	}
}