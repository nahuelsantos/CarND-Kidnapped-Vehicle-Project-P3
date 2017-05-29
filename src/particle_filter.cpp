/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 150;
	weights.resize(num_particles);	
	
	for (int i = 0; i < num_particles; i++) 
	{
		random_device rd;
		default_random_engine dre(rd());
		normal_distribution<double> gps_error_x(x, std[0]);
		normal_distribution<double> gps_error_y(y, std[1]);
		normal_distribution<double> gps_error_theta(theta, std[2]);

        double x = gps_error_x(dre);
        double y = gps_error_y(dre);
        double theta = gps_error_theta(dre);
		double weight = 1.0;

		Particle particle = { i, x, y, theta, weight };
		particles.push_back(particle);

		weights[i] = 1.0;
	}
	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	random_device rd;
	default_random_engine dre(rd());	
	
	for (int i = 0; i < num_particles; i++) 
	{
		if (fabs(yaw_rate) > 1e-4)
		{
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
			particles[i].theta += yaw_rate * delta_t;
		} 
		else 
		{
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		
		normal_distribution<double> error_x(particles[i].x, pow(std_pos[0], 2));
		normal_distribution<double> error_y(particles[i].y, pow(std_pos[1], 2));
		normal_distribution<double> error_theta(particles[i].theta, pow(std_pos[2], 2));

		particles[i].x = error_x(dre);
		particles[i].y = error_y(dre);
		particles[i].theta = error_theta(dre);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	int observations_size = observations.size();
	int predicted_size = predicted.size();

	for (int i = 0; i < observations_size; i++) 
	{
		double min_distance = 99999.9;

		for (int j = 0; j < predicted_size; j++) 
		{
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			if (distance < min_distance) 
			{
				min_distance = distance;
				observations[i].id = predicted[j].id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	weights.clear();

	for (int i = 0; i < num_particles; i++) 
	{
		double current_x = particles[i].x;
		double current_y = particles[i].y;
		double current_theta = particles[i].theta;

		vector<LandmarkObs> observations_map;
		int observations_size = observations.size();
		    
	    for (int j = 0; j < observations_size; j++)
		{
			LandmarkObs landmark;
			landmark.x = observations[j].x * cos(current_theta) - observations[j].y * sin(current_theta) + current_x;
			landmark.y = observations[j].x * sin(current_theta) + observations[j].y * cos(current_theta) + current_y;

			observations_map.push_back(landmark);
		}

		vector<LandmarkObs> predicted_landmarks;
		int landmarks_size = map_landmarks.landmark_list.size();

		for (int j = 0; j < landmarks_size; j++)
		{
			double distance = dist(current_x, current_y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
			
		 	if (distance <= sensor_range)
		 	{
				LandmarkObs landmark;
				landmark.id = map_landmarks.landmark_list[j].id_i;
				landmark.x = map_landmarks.landmark_list[j].x_f;
				landmark.y = map_landmarks.landmark_list[j].y_f;

			 	predicted_landmarks.push_back(landmark);
		 	}
		}

		dataAssociation(predicted_landmarks, observations_map);
		 
		double weight = 1.0;
		int predicted_landmarks_size = predicted_landmarks.size();

		for (int j = 0; j < predicted_landmarks_size; j++)
		{
			int min_index = -1;
			double min_distance = 99999.9;

			for (int k = 0; k < observations_map.size(); k++)
			{
				if (predicted_landmarks[j].id == observations_map[k].id )
				{
					double distance = dist(predicted_landmarks[j].x, predicted_landmarks[j].y, observations_map[k].x, observations_map[k].y);

					if (distance <= min_distance){
						min_index = k;
						min_distance = distance;
					}
				 }
			 }
			 
			 double delta_x = predicted_landmarks[j].x - observations_map[min_index].x;
			 double delta_y = predicted_landmarks[j].y - observations_map[min_index].y;

			 weight *= exp(-0.5 * (pow(delta_x, 2.0) * std_landmark[0] + pow(delta_y, 2.0) * std_landmark[1])) / sqrt(2.0 * M_PI * std_landmark[0] * std_landmark[1]);
		 }
		 
		 weights.push_back(weight);
		 particles[i].weight = weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	random_device rd;
	default_random_engine dre(rd());		

	discrete_distribution<int> distribution_weights(weights.begin(), weights.end());
	vector<Particle> new_particles;

	for(int i = 0; i < num_particles; i++) 
	{
		Particle particle = particles[distribution_weights(dre)];
		new_particles.push_back(particle);
	}
	
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
