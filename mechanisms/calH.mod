TITLE HVA (L-type) calcium channel with low threshold for activation (dendrites)

COMMENT
Used in distal dendrites to account for distally restricted initiation of Ca++ spikes.
Uses channel conductance (not permeability).
Written by Yiota Poirazi, 1/8/00 poirazi@LNC.usc.edu.
ENDCOMMENT

NEURON {
	SUFFIX calH
	USEION ca READ eca WRITE ica
	RANGE gcalbar, ica
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {	:parameters that can be entered when function is called in cell-setup
	v (mV)
	celsius = 34	(degC)
	gcalbar = 0 (mho/cm2)	: initialized conductance
	eca = 140 (mV)	: Ca2+ reversal potential
}

ASSIGNED {	:parameters needed to solve DE
	ica (mA/cm2)
	minf (1)
	hinf (1)
	taum (ms)
	tauh (ms)
}

STATE {	:unknown activation and inactivation parameters to be solved in the DEs
	m
	h
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	ica = gcalbar*pow(m, 3)*h*(v - eca)
}

DERIVATIVE states {
	rates (v)
	m' = (minf-m)/taum
	h' = (hinf-h)/tauh
}

INITIAL {
	m = 0	:initial activation parameter value
	h = 1	:initial inactivation parameter value
	rates(v)
	ica = gcalbar*m*m*m*h*(v - eca) : initial Ca++ current value
}

PROCEDURE rates(v (mV)) {
	minf = 1 / (1 + exp(-(v+37(mV))/1(mV)))		:Ca activation 
	taum = 3.6

	hinf = 1 / (1 + exp((v+41(mV))/0.5(mV)))	:Ca inactivation
	tauh = 29
}