#include <cstdio>
#include <memory>
#include <thread>
#include <iostream>

// -------------- HOPFIELD NETWORKS --------------
#include "hopfield/states/binary.hpp"
#include "hopfield/deterministic/dense_hopfield_network.hpp"
#include "hopfield/stochastic/stochastic_hopfield_network.hpp"
#include "hopfield/deterministic/cross_talk_visualizer.hpp"

#include "hopfield/logger/logger.hpp"

// -------------- INPUT ROUTINES --------------
#include "io/datasets/dataset.hpp"
#include "io/datasets/repository.hpp"
#include "io/image/images.hpp"

int main() {

	using Hebbs = HebbianPolicy<float>;

	// Create a plotter object to plot what we require. 
	Plotter p;

	constexpr const auto SIZE = 128;

	std::cout << "Creo" << std::endl;
	DenseHopfieldNetwork<Hebbs>				dhn(SIZE * SIZE);
	std::cout << "Creato!" << std::endl;

	Image rose_img("rose.png", Channels::Greyscale);

	ImageUtils::niblack_binarize(rose_img, 10);

	// Load all the binary states with the image representatives. 
	BinaryState rose(SIZE * SIZE);
	StateUtils::load_state_from_byte_array(rose, rose_img.data(), SIZE * SIZE);

	BinaryState deformed_rose(SIZE * SIZE);
	deformed_rose.copy_content(rose);
	StateUtils::perturb_state(deformed_rose, /* Noise level */ 0.25);

	BinaryState deformed2(SIZE * SIZE);
	deformed2.copy_content(rose);
	StateUtils::perturb_state(deformed2, /* Noise level */ 0.55);

	BinaryState deformed3(SIZE * SIZE);
	deformed3.copy_content(rose);
	StateUtils::perturb_state(deformed3, /* Noise level */ 0.3);
	BinaryState deformed4(SIZE * SIZE);
	deformed4.copy_content(rose);
	StateUtils::perturb_state(deformed4, /* Noise level */ 0.14);

	BinaryState deformed5(SIZE * SIZE);
	deformed5.copy_content(rose);
	StateUtils::perturb_state(deformed5, /* Noise level */ 0.10);

	BinaryState deformed6(SIZE * SIZE);
	deformed6.copy_content(rose);
	StateUtils::perturb_state(deformed6, /* Noise level */ 0.8);


	deformed_rose.set_stride_y(SIZE);
	dhn.set_state_strides(SIZE);
	StateUtils::plot_state(p, deformed_rose);

	rose.set_stride_y(SIZE);

	StateUtils::plot_state(p, rose);
	p.block();
	std::cout << "> Storing the patterns in the networks: " << std::endl;

	dhn.store(rose);

	std::cout << "> Initializing the logger: " << std::endl;
	// Create a logger object to visualize the action of the network over time. 
	auto logger = HopfieldLogger(&p);
	std::ofstream save_file("save_data.txt");
	{
		// Setup all configurations for the logger
		logger.set_recording_stream(save_file, true);

		logger.set_collect_states(true, "parallel_deterministic_states.gif");
		logger.set_collect_energy(true);
		logger.set_collect_temperature(true);

		// We collect order parameters, which represent averages in time
		// of the agreement between the state of the network and other patterns. 
		logger.set_collect_order_parameter(true);
		dhn.add_reference_state(rose);
		dhn.add_reference_state(deformed2);
		dhn.add_reference_state(deformed3);
		dhn.add_reference_state(deformed4);
		dhn.add_reference_state(deformed5);
		dhn.add_reference_state(deformed6);


		logger.set_prefix("Deterministic network");
		logger.finally_write_last_state_png(true, "parallel_deterministic_last_state.png");
		logger.finally_plot_data(true);
	}
	dhn.attach_logger(&logger);

	// --- Now display the denoising property of the hopfield network ---
	dhn.set_state_strides(SIZE);

	UpdateConfig uc = { UpdatePolicy::GroupUpdate, int((512.0 / 100) * 1 * 512) };

	std::cout << "> Feeding the pattern to the network: " << std::endl;
	dhn.feed(deformed_rose);
	std::cout << "> Running the deterministic network: " << std::endl;
	dhn.run(/* Iterations */ 20, uc);
	// Implicit plotting... because of the logger settings. 

	p.block();

}