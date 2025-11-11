TITLE HVAm (R-type) calcium (CA2+) channel with medium threshold for activation

COMMENT
Used in somatic regions.
It has lower threshold for activation/inactivation and slower activation time constant,
than the same mechanism in dendritic regions.
Uses channel conductance (not permeability).
Written by Yiota Poirazi on 3/12/01 poirazi@LNC.usc.edu.
ENDCOMMENT

NEURON {
	SUFFIX somacar
	USEION ca READ eca WRITE ica
	RANGE gcabar, ica
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER { : parameters that can be entered when function is called in cell-setup
	v (mV)
	celsius = 34	(degC)
	gcabar = 0 (mho/cm2) : initialized conductance
	eca = 140 (mV)	: Ca2+ reversal potential
}

STATE { : unknown activation and inactivation parameters to be solved in the DEs
	m
	h
}

ASSIGNED { : parameters needed to solve DE
	ica (mA/cm2)
	minf (1)
	hinf (1)
	taum (ms)
	tauh (ms)
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	ica = gcabar*pow(m, 3)*h*(v - eca)
}

DERIVATIVE states {
	rates(v)
	m' = (minf - m)/taum
	h' = (hinf - h)/tauh
}

INITIAL {
	m = 0	: initial activation parameter value
	h = 1	: initial inactivation parameter value
	rates(v)
	ica = gcabar*pow(m, 3)*h*(v - eca)
}

PROCEDURE rates(v (mV)) {
	minf = 1 / (1 + exp((v + 60(mV))/(-3(mV))))	:Ca activation
	hinf = 1 / (1 + exp((v + 62(mV))/(1(mV))))	:Ca inactivation
	taum = 100.0
	tauh = 5.0
}