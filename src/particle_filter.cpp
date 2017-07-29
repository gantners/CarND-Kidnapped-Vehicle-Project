/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */
#define M_PI 3.14159265358979323846

#include <random>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>

#include "particle_filter.h"

using namespace std;

/*
 *  //Set up parameters here
  double delta_t = 0.1; // Time elapsed between measurements [sec]
  double sensor_range = 50; // Sensor range [m]

  double sigma_pos [3] = {0.3, 0.3, 0.01}; // GPS measurement uncertainty [x [m], y [m], theta [rad]]
  double sigma_landmark [2] = {0.3, 0.3}; // Landmark measurement uncertainty [x [m], y [m]]
 * */

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    /*
    int id;
    double x;
    double y;
    double theta;
    double weight;
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;
     */

    num_particles = 100;

    default_random_engine gen;

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_psi(theta, std[2]);

    for (int i = 0; i < num_particles; ++i) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen); //gaussian noise
        p.y = dist_y(gen);//gaussian noise
        p.theta = dist_psi(gen);//gaussian noise
        p.weight = 1.0;
        particles.push_back(p);
        weights.push_back(1.0);
    }
    weights = std::vector<double>(num_particles);
    is_initialized = true;
}

// Predict the vehicle's next state from previous (noiseless control) data.
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;

    for (int i = 0; i < particles.size(); ++i) {
        Particle &p = particles[i];

        //avoid division through zero
        if (yaw_rate == 0.0) {
            p.x = p.x + velocity * delta_t * cos(p.theta);
            p.y = p.y + velocity * delta_t * sin(p.theta);

        } else {
            p.x = p.x + (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
            p.y = p.y + (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
            p.theta = p.theta + yaw_rate * delta_t;
        }
        normal_distribution<double> dist_x(p.x, std_pos[0]);
        normal_distribution<double> dist_y(p.y, std_pos[1]);
        normal_distribution<double> dist_psi(p.theta, std_pos[2]);

        p.x = dist_x(gen); //gaussian noise
        p.y = dist_y(gen);//gaussian noise
        p.theta = dist_psi(gen);//gaussian noise
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    // << "Predicted size: " << predicted.size() << endl;
    //cout << "Observations size: " << observations.size() << endl;
    for (int oi = 0; oi < observations.size(); ++oi) {
        LandmarkObs &o = observations[oi];
        int closestId;
        double minimum = 99999;

        for (int pi = 0; pi < predicted.size(); ++pi) {
            LandmarkObs &p = predicted[pi];
            double d = dist(o.x, o.y, p.x, p.y); //distance between landmark in range and observed spot
            if (d < minimum) { // if closer than current set new closest landmark
                minimum = d;
                closestId = p.id;
            }
        }
        //cout << "closest LM id for observation id " << o.id << " = " << closestId << endl;
        //Closest landmark id for transformed observation
        o.id = closestId;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a multi-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    // in respect to each particle
    for (int i = 0; i < num_particles; ++i) {
        Particle &p = particles[i];

        vector<LandmarkObs> transformed;

        //for each observation
        for (int oi = 0; oi < observations.size(); ++oi) {
            LandmarkObs &o = observations[oi];
            double x_prime;
            double y_prime;

            //transform into map coordinates in respect to our current particle using a single function
            x_prime = p.x + o.x * cos(p.theta) - o.y * sin(p.theta);
            y_prime = p.y + o.x * sin(p.theta) + o.y * cos(p.theta);
            //cout << "OBS," << oi <<  "," <<x_prime << "," << y_prime << endl;
            transformed.push_back(LandmarkObs{oi, x_prime, y_prime});
        }

        vector<LandmarkObs> predictions;

        //Get all landmarks in sensor range in respect to particle
        for (int i = 0; i < map_landmarks.landmark_list.size(); i++) {
            Map::single_landmark_s lm = map_landmarks.landmark_list[i];
            double d = fabs(dist(lm.x_f, lm.y_f, p.x, p.y)); // distance of particle to landmark
            if (d <= sensor_range) { // if particle in range of landmark, we have a possible car location
                //cout << "LM," << lm.id_i <<  "," << lm.x_f << "," << lm.y_f << ", in range of particle ," << p.id  << "," << p.x << "," << p.y << endl;
                predictions.push_back(LandmarkObs{i, (double) lm.x_f, (double) lm.y_f});
            }
        }




        // associate each transformed observation with the nearest landmark in sensor range
        //cout << "Landmark size: " << map_landmarks.landmark_list.size() << endl;
        dataAssociation(predictions, transformed);
        //now we added for each observation the closed lm id

        double new_weight = 1.0;

        //update weights
        for (LandmarkObs &t : transformed) {
            Map::single_landmark_s nearest =  map_landmarks.landmark_list[t.id];
            double x2 = pow(t.x - (double) nearest.x_f, 2) / (2.0 * pow(std_landmark[0], 2));
            double y2 = pow(t.y - (double) nearest.y_f, 2) / (2.0 * pow(std_landmark[1], 2));
            double probability = 1 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]) * exp(-(x2 + y2));
            new_weight *= probability; // product of all weights of this particle
        }

        //set new particle weight
        p.weight = new_weight;
        // Remember weight for normalizing later
        weights[i] = new_weight;
    }

    //normalize weights
    double weight_sum = accumulate(weights.begin(), weights.end(), 0.0);
    for (int i = 0; i < num_particles; ++i) {
        particles[i].weight /= weight_sum;
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;
    discrete_distribution<int> disc(weights.begin(), weights.end());
    vector<Particle> resampled;
    for (int i = 0; i < particles.size(); ++i) {
        resampled.push_back(particles[disc(gen)]);
    }
    particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
