#include "hmm.h"
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct GMM {
	double weight;
	std::vector<double> mean, var;
};

struct HMMState {
	std::vector<GMM> pdf;
};

struct HMM {
	std::vector<std::vector<double>> tp;
	std::vector<HMMState> states;
};

struct WordHMM : HMM {
	std::string name;
};

struct FullHMM : HMM {
	std::map<int, std::string> wordStarts;
};

void initMatrix(std::vector<std::vector<double>> &m, int rows, int cols) {
	std::vector<double> row(cols, 0.0);
	for (int i = 0; i < rows; i++) {
		m.push_back(row);
	}
}

void initMatrix(std::vector<std::vector<double>> &m, int size) {
	initMatrix(m, size, size);
}

void copyMatrix(
	const std::vector<std::vector<double>> &src,
	int srcRow, int srcCol,
	int rows, int cols,
	std::vector<std::vector<double>> &dst,
	int dstRow, int dstCol) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			dst[dstRow + i][dstCol + j] = src[srcRow + i][srcCol + j];
		}
	}
}

void printMatrix(const std::vector<std::vector<double>> &m) {
	std::cout << "size: " << m.size() << std::endl;
	for (auto &row : m) {
		for (auto col : row) {
			std::cout << col << ' ';
		}
		std::cout << std::endl;
	}
}

WordHMM concatPhones(const std::vector<HMM> &models) {
	WordHMM result;
	int Ns = N_STATE * models.size();
	initMatrix(result.tp, Ns + 2);
	result.tp[0][1] = 1.0;
	int offset = 1;
	for (auto &model : models) {
		for (auto state : model.states) {
			result.states.push_back(state);
		}
		copyMatrix(model.tp, 1, 1, N_STATE, N_STATE + 1, result.tp, offset, offset);
		offset += N_STATE;
	}
	return result;
}

WordHMM concatSp(const WordHMM &model, const HMM &sp) {
	WordHMM result;
	for (auto state : model.states)
		result.states.push_back(state);
	for (auto state : sp.states)
		result.states.push_back(state);
	initMatrix(result.tp, result.states.size() + 2);

	int Ns = model.states.size();
	float exitProb = model.tp[Ns][Ns + 1];
	copyMatrix(model.tp, 0, 0, Ns + 2, Ns + 2, result.tp, 0, 0);
	copyMatrix(sp.tp, 0, 1, 2, 2, result.tp, Ns, Ns + 1);
	result.tp[Ns][Ns + 1] *= exitProb;
	return result;
}

FullHMM makeBigram(const std::vector<WordHMM> &models, std::map<std::string, std::map<std::string, double> > &bigrams) {
	FullHMM result;
	std::vector<int> enterStates;
	for (auto &model : models) {
		enterStates.push_back(result.states.size() + 1);
		for (auto state : model.states)
			result.states.push_back(state);
	}

	int Ns = result.states.size();
	initMatrix(result.tp, Ns + 2);
	for (int i = 0; i < models.size(); i++) {
		auto &model = models[i];
		int enter = enterStates[i];
		int s = model.states.size();
		int exit = enter + s - 1;
		result.tp[0][enter] = bigrams["<s>"][model.name];
		copyMatrix(model.tp, 1, 1, s, s, result.tp, enter, enter);
		for (int j = 0; j < models.size(); j++) {
			auto &model2 = models[j];
			result.tp[exit][enterStates[j]] = bigrams[model.name][model2.name];
		}
		result.tp[exit][Ns + 1] = bigrams[model.name]["<s>"];
		result.wordStarts[enter - 1] = model.name;
	}
	return result;
}

FullHMM prependSil(const FullHMM &model, const HMM &sil) {
	FullHMM result;
	for (auto state : sil.states)
		result.states.push_back(state);
	for (auto state : model.states)
		result.states.push_back(state);

	// shift word start index
	for (auto pair : model.wordStarts)
		result.wordStarts[pair.first + sil.states.size()] = pair.second;

	int Ns = result.states.size();
	initMatrix(result.tp, Ns + 2);
	copyMatrix(sil.tp, 0, 0, N_STATE + 2, N_STATE + 2, result.tp, 0, 0);
	int s = model.states.size();
	copyMatrix(model.tp, 0, 1, s + 2, s + 1, result.tp, N_STATE, N_STATE + 1);

	double silExitProb = sil.tp[N_STATE][N_STATE + 1];
	for (int j = N_STATE + 1; j < Ns + 2; j++) {
		result.tp[N_STATE][j] *= silExitProb;
	}
	return result;
}

FullHMM buildFullModel() {
	std::map<std::string, HMM> phones;
	for (auto hmm : data) {
		HMM model;
		int Ns = N_STATE;
		if (hmm.name == std::string("sp")) {
			Ns = 1;
		}
		initMatrix(model.tp, Ns + 2);
		for (int i = 0; i < Ns + 2; i++) {
			for (int j = 0; j < Ns + 2; j++) {
				model.tp[i][j] = hmm.tp[i][j];
			}
		}
		for (int i = 0; i < Ns; i++) {
			HMMState state;
			for (int j = 0; j < N_PDF; j++) {
				GMM pdf;
				pdf.weight = hmm.state[i].pdf[j].weight;
				for (int k = 0; k < N_DIMENSION; k++) {
					pdf.mean.push_back(hmm.state[i].pdf[j].mean[k]);
					pdf.var.push_back(hmm.state[i].pdf[j].var[k]);
				}
				state.pdf.push_back(pdf);
			}
			model.states.push_back(state);
		}
		phones[hmm.name] = model;
	}

	std::vector<WordHMM> words;
	{
		std::ifstream f("data/dictionary.txt");
		while (!f.eof()) {
			std::string line;
			std::getline(f, line);
			std::stringstream ss(line);
			std::string name;
			ss >> name;
			std::string phoneName;
			std::vector<HMM> phoneSeq;
			while (ss >> phoneName) {
				phoneSeq.push_back(phones[phoneName]);
			}
			WordHMM model = concatPhones(phoneSeq);
			model = concatSp(model, phones["sp"]);
			model.name = name;
			words.push_back(model);
		}
	}

	std::map<std::string, std::map<std::string, double>> bigrams;
	{
		std::ifstream f("data/bigram.txt");
		while (!f.eof()) {
			std::string line;
			std::getline(f, line);
			std::stringstream ss(line);
			std::string first, second;
			double prob;
			ss >> first >> second >> prob;
			bigrams[first][second] = prob;
		}
	}

	return prependSil(makeBigram(words, bigrams), phones["sil"]);
}

double getStateLogPdf(const HMMState &state, const std::vector<double> &xs) {
	double maxp = -std::numeric_limits<double>::infinity();
	for (auto &pdf : state.pdf) {
		double logp = log(pdf.weight);
		for (int j = 0; j < N_DIMENSION; j++) {
			//exp(-(x - mean[i]) ** 2 / (2 * var[i])) / sqrt(2 * pi * var[i])
			logp += -pow(xs[j] - pdf.mean[j], 2) / (2.0 * pdf.var[j]) - log(2.0 * M_PI * pdf.var[j]) / 2.0;
		}
		if (logp > maxp)
			maxp = logp;
	}
	return maxp;
}

int main() {
	FullHMM hmm = buildFullModel();

	std::vector<std::vector<double> > obs;
	{
		std::ifstream f("tst/f/ak/84z3z51.txt");
		int rows, cols;
		f >> rows >> cols;
		for (int i = 0; i < rows; i++) {
			std::vector<double> row;
			for (int j = 0; j < cols; j++) {
				double n;
				f >> n;
				row.push_back(n);
			}
			obs.push_back(row);
		}
	}

	int Ns = hmm.states.size();
	int T = obs.size();
	std::vector<std::vector<double> > delta, psi;
	initMatrix(delta, T, Ns);
	initMatrix(psi, T, Ns);

	for (int i = 0; i < Ns; i++) {
		delta[0][i] = log(hmm.tp[0][1 + i]) + getStateLogPdf(hmm.states[i], obs[0]);
	}

	for (int t = 1; t < T; t++) {
		for (int j = 0; j < Ns; j++) {
			double maxp = -std::numeric_limits<double>::infinity();
			int maxi = 0;
			for (int i = 0; i < Ns; i++) {
				double p = delta[t - 1][i] + log(hmm.tp[1 + i][1 + j]);
				if (p > maxp) {
					maxp = p;
					maxi = i;
				}
			}
			delta[t][j] = maxp + getStateLogPdf(hmm.states[j], obs[t]);
			psi[t][j] = maxi;
		}
	}

	// backtracking

	double maxp = -std::numeric_limits<double>::infinity();
	int maxi = 0;
	for (int i = 0; i < Ns; i++) {
		double p = delta[T - 1][i];
		if (p > maxp) {
			maxp = p;
			maxi = i;
		}
	}

	std::vector<std::string> result;

	int prev = maxi;
	for (int t = T - 2; t >= 0; t--) {
		int i = psi[t + 1][prev];
		if (i != prev && i != prev - 1) {
			result.push_back(hmm.wordStarts[prev]);
		}
		prev = i;
	}

	std::reverse(result.begin(), result.end());

	for (auto word : result)
		std::cout << word << std::endl;
	
	return 0;
}
