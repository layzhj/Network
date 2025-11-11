TITLE HVAm (R-type) calcium (CA2+) channel with medium threshold for activation

COMMENT
Used in distal dendritic regions, together with calH.mod,
to help the generation of Ca++ spikes in these regions.
Uses channel conductance (not permeability).
Written by Yiota Poirazi on 11/13/00 poirazi@LNC.usc.edu.
ENDCOMMENT

NEURON {
	SUFFIX car
	USEION ca READ eca WRITE ica
	RANGE gcabar, ica
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	: parameters that can be entered when function is called in cell-setup
	v (mV)
	celsius = 34	(degC)
	gcabar = 0 (mho/cm2) : initialized conductance
	eca = 140 (mV) 		: Ca2+ reversal potential, 140 (mV)
}

STATE {	m h } : unknown activation and inactivation parameters to be solved in the DEs

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
	minf = 1 / (1 + exp(-(v+48.5(mV))/3(mV)))	: Ca activation
	taum = 50

	hinf = 1 / (1 + exp((v+53(mV))/(1(mV))))	: Ca inactivation
	tauh = 5
}