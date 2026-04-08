#ifnedf SYNAPSES_HPP
#define SYNAPSES_HPP

enum SynapseType {
	// Kunihiko Fukushima, "Cognitron: A Self-organizing Multilayered Neural Network"
	// Finds three different kinds of synapses, each of which evolves differently. The 
	// purpose is to emulate the dozens of kinds of neurons in our brain. 
	Hebb, 
	Brindley,
	Modifiable
};


#endif