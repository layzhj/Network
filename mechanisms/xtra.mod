: $Id: modified version of Ted's xtra.mod, which allows for extracellular recording without the extracellular mechanism $

COMMENT
Now uses i_membrane_ (see CVode class's use_fast_imem documentation)
See  https://www.neuron.yale.edu/phpBB/viewtopic.php?f=2&t=3389&p=14342&hilit=extracellular+recording+parallel#p14342
ENDCOMMENT

NEURON {
	SUFFIX xtra
	RANGE rx, er
	RANGE x, y, z
:	GLOBAL is
:	POINTER im, ex
	POINTER im
}

PARAMETER {
	: default transfer resistance between stim electrodes and axon
	rx = 1 (megohm) : mV/nA
	x = 0 (1) : spatial coords
	y = 0 (1)
	z = 0 (1)
}

ASSIGNED {
	v (millivolts)
:	is (milliamp)
:	ex (millivolts)
:	im (milliamp/cm2)
	im (nanoamp)
	er (microvolts)
:	area (micron2)
}

INITIAL {
:	ex = is*rx*(1e6)
:	er = (10)*rx*im*area
	er = (1000)*rx*im
: this demonstrates that area is known
: UNITSOFF
: printf("area = %f\n", area)
: UNITSON
}

: Use BREAKPOINT for NEURON 5.4 and earlier
: BREAKPOINT {
:	SOLVE f METHOD cvode_t
: }
:
: PROCEDURE f() {
:	: 1 mA * 1 megohm is 1000 volts
:	: but ex is in mV
:	ex = is*rx*(1e6)
:	er = (10)*rx*im*area
: }

: With NEURON 5.5 and later, abandon the BREAKPOINT block and PROCEDURE f(),
: and instead use BEFORE BREAKPOINT and AFTER BREAKPOINT

:BEFORE BREAKPOINT { : before each cy' = f(y,t) setup
:  ex = is*rx*(1e6)
:}
AFTER SOLVE { : after each solution step
:  er = (10)*rx*im*area
   er = (1000)*rx*im
}