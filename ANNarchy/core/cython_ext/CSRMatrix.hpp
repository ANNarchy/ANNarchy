/*
 *  Copyright (C) 2016-2018 Helge Dinkelbach <helge.dinkelbach@gmail.com>
 *
 *  CSRMatrix.hpp is part of ANNarchy
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Foobar is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this headers. If not, see <http://www.gnu.org/licenses/>.
 */
#pragma once
#include <iostream>
#include <random>
#include <iomanip>

template<typename VT = double>
class CSRMatrix {
	const int num_rows_;            ///< number of rows

	std::vector<int> fwd_row_;      ///< i-th row in the matrix ranges from fwd_row_[i] to fwd_row_[i+1]
	std::vector<int> fwd_col_idx_;  ///< accessing with fwd_col_idx_[fwd_row_[i]] to fwd_col_idx_[fwd_row_[i+1]] provides acces to all column indices
	std::vector<VT> values_;    ///< weight values
	std::vector<int> delays_;       ///< delays in computation steps ( not time! )

public:
	/**
	 * 	\brief 		constructor
	 */
	CSRMatrix(const int num_rows) :
		num_rows_(num_rows+1) {

		fwd_row_ = std::vector<int>(num_rows_, 0);
		fwd_col_idx_ = std::vector<int>();
		values_ = std::vector<VT>();
		delays_ = std::vector<int>();
	}

	/**
	 * 	\brief 		destructor
	 */
	~CSRMatrix() {
		fwd_row_.clear();
		fwd_col_idx_.clear();
		values_.clear();
		delays_.clear();
	}

	/**
	 * 	\brief		get size in bytes allocated by the CSRMatrix
	 */
	inline size_t size_in_bytes() {
		size_t size = 0;

		size += fwd_row_.capacity() * sizeof(int);
		size += fwd_col_idx_.capacity() * sizeof(int);
		size += values_.capacity() * sizeof(VT);
		size += delays_.capacity() * sizeof(int);

		return size;
	}

	inline int num_rows() {
		return num_rows_-1;
	}

	inline std::vector<int> row_begin() {
		return fwd_row_;
	}

	inline std::vector<int> column_indices() {
		return fwd_col_idx_;
	}

	inline std::vector<VT> values() {
		return values_;
	}

	inline std::vector<int> delays() {
		return delays_;
	}

	inline int num_elements() {
		return fwd_col_idx_.size();
	}

	/**
	 * 	\brief		add all connections of one dendrite
	 * 	\details
	 * 	\param[in]	row			neuron index of afferent neuron
	 * 	\param[in]	columns		neuron indices of efferent neurons
	 * 	\param[in]	w			synaptic weights
	 */
	void push_back(int row, std::vector<int> columns, std::vector<VT> w, std::vector<int> d) {
	#ifdef _DEBUG
		std::cout << "push_back: row = " << row << ", columns.size() = " << columns.size() << ", w.size() = " << w.size() << ", d.size() = " << d.size() << std::endl;
	#endif
		int old_idx = fwd_row_[row+1];

		// column indices and update row_ptr
		fwd_col_idx_.insert(fwd_col_idx_.begin()+old_idx, columns.begin(), columns.end());
        for( auto it = fwd_row_.begin()+row+1; it != fwd_row_.end(); it++ )
            *it += columns.size();

		// synaptic weights
		if (w.size() > 1 || columns.size() == 1) {
			// either multiple weights, or a single connection
			values_.insert(values_.begin()+old_idx, w.begin(), w.end());
		} else {
			auto tmp_w = std::vector<VT>(columns.size(), w[0]);
			values_.insert(values_.begin()+old_idx, tmp_w.begin(), tmp_w.end());
		}

		// synaptic delays
        if (d.size() > 1 || columns.size() == 1) {
            // either multiple weights, or a single connection
            delays_.insert(delays_.begin()+old_idx, d.begin(), d.end());
        } else {
            auto tmp_d = std::vector<int>(columns.size(), d[0]);
            delays_.insert(delays_.begin()+old_idx, tmp_d.begin(), tmp_d.end());
        }
	}

	/**
	 * 	\brief		overloaded std::ostream operator<<
	 * 	\details	for the object itself
	 * 	\param[IN]	os		ostream instance
	 * 	\param[IN]	matrix	object instance
	 * 	\return		manipulated ostream instance
	 */
	friend std::ostream& operator<< (std::ostream& os, const CSRMatrix& matrix) {
        os << "(forward)" << std::endl;
		os << "fwd_row_:" << std::endl;
		os << "[ ";
		for(int r = 0; r < matrix.num_rows_; r++) {
			os << matrix.fwd_row_[r] << " ";
		}
		os << "]" << std::endl;

		os << "fwd_col_idx:" << std::endl;
		os << "[ ";
		for(unsigned int idx = 0; idx < matrix.values_.size(); idx++) {
			os << matrix.fwd_col_idx_[idx] << " ";
		}
		os << "]" << std::endl;

		os << "values:" << std::endl;
		os << "[ ";
		for(unsigned int idx = 0; idx < matrix.values_.size(); idx++) {
			os << std::setprecision(2) << matrix.values_[idx] << " ";
		}
		os << "]" << std::endl;

		os << "delays:" << std::endl;
		os << "[ ";
		for(unsigned int idx = 0; idx < matrix.delays_.size(); idx++) {
			os << std::setprecision(2) << matrix.delays_[idx] << " ";
		}
		os << "]" << std::endl;

		return os;
	}

	/**
	 * 	\brief		overloaded std::ostream operator<<
	 * 	\details	for the reference to an object
	 * 	\param[IN]	os		ostream instance
	 * 	\param[IN]	matrix	object reference
	 * 	\return		manipulated ostream instance
	 */
	friend std::ostream& operator<< (std::ostream& os, CSRMatrix* matrix) {
		return os << *matrix;
	}
};
