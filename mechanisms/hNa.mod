TITLE  H-current that uses Na ions (Poirazi)

NEURON {
	SUFFIX hNa
    RANGE  gbar,vhalf, K, taun, ninf
	USEION na READ ena WRITE ina      
:	NONSPECIFIC_CURRENT i
}

UNITS {
	(um) = (micrometer)
	(mA) = (milliamp)
	(uA) = (microamp)
	(mV) = (millivolt)
	(pmho) = (picomho)
	(mmho) = (millimho)
}

PARAMETER {              : parameters that can be entered when function is called in cell-setup
    ena    = 50    (mV)
    eh     = -10   (mV)
	K      = 8.5   (mV)
:	gbar   = 0.1   (mmho/cm2) : suggested somatic value, the dendritic value is ~6x higher
	gbar   = 0     (mho/cm2)  : initialize conductance to zero
	vhalf  = -90   (mV)       : half potential
}	


STATE {                : the unknown parameters to be solved in the DEs
	n
}

ASSIGNED {             : parameters needed to solve DE
	ina (mA/cm2)
	ninf
	taun (ms)
	v (mV)
}

        


INITIAL {               : initialize the following parameter using states()
	rates(v)	
	n = ninf
	ina = gbar*n*(v-eh)*0.001            :0.001 used to fix units of g (given in mmho/cm2 to mho/cm2)
}


BREAKPOINT {
	SOLVE states METHOD derivimplicit
:	ina = gbar*n*(v-ena)*(0.001)   :0.001 used to fix units of g (given in mmho/cm2 to mho/cm2)
	ina = gbar*n*(v-eh)*0.001            :0.001 used to fix units of g (given in mmho/cm2 to mho/cm2)
}

DERIVATIVE states {
	rates(v)
    n' = (ninf - n)/taun
}

PROCEDURE rates(v (mV)) {  
 
 	if (v > -30) {
		taun = 1
	} else {
           taun = 2(ms)*(1/(exp((v+145(mV))/-17.5(mV))+exp((v+16.8(mV))/16.5(mV))) + 5) :h activation tau

	}  
    ninf = 1 - (1 / (1 + exp((vhalf - v)/K)))                  :steady state value
}



