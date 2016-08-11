/*
 *  Copyright (C) 2016-2018 Helge Dinkelbach <helge.dinkelbach@gmail.com>
 *
 *  DenseMatrix.hpp is part of a set of headers. If you use it in your work,
 *  you may refer to TODO: article
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

class CSRCMatrix {
	const unsigned int num_rows_;					///< number of rows
	const unsigned int num_columns_;				///< number of columns

	std::vector<unsigned int> row_idx_;				///< indices of non-zero columns
	std::vector<unsigned int> col_idx_;				///< indices of non-zero rows

	std::vector<unsigned int> fwd_row_;		        ///< i-th row in the matrix ranges from fwd_row_[i] to fwd_row_[i+1]
	std::vector<unsigned int> fwd_col_idx_;         ///< accessing with fwd_col_idx_[fwd_row_[i]] to fwd_col_idx_[fwd_row_[i+1]] provides acces to all column indices
	std::vector<double> values_;

	std::vector<unsigned int> bwd_col_;             ///< j-th column in the matrix ranges from bwd_col_[i] to bwd_col_[i+1]
	std::vector<unsigned int> bwd_row_idx_;
	std::vector<unsigned int> bwd_inv_idx_;


public:
	CSRCMatrix(const unsigned int num_rows, const unsigned int num_columns) :
		num_rows_(num_rows+1), num_columns_(num_columns+1) {

		row_idx_ = std::vector<unsigned int>();

		fwd_row_ = std::vector<unsigned int>(num_rows_, 0);
		fwd_col_idx_ = std::vector<unsigned int>();
		values_ = std::vector<double>();

		bwd_col_ = std::vector<unsigned int>(num_columns_, 0);
		bwd_row_idx_ = std::vector<unsigned int>();
		bwd_inv_idx_ = std::vector<unsigned int>();
	}

	~CSRCMatrix() {
		// TODO:
	}

	inline size_t size_in_bytes() {
		size_t size = 0;

		size += values_.size() * sizeof(unsigned int); /* col_idx */
		size += values_.size() * sizeof(unsigned int); /* row_idx */
		size += values_.size() * sizeof(unsigned int); /* inv_idx */

		size += values_.size() * sizeof(double); /* values */

		size += (num_rows_) * sizeof(unsigned int); /* row_begin */
		size += (num_columns_) * sizeof(unsigned int); /* col_begin */

		return size;
	}

	inline unsigned int num_rows() {
		return num_rows_-1;
	}

	inline unsigned int num_columns() {
		return num_columns_-1;
	}

	inline double* values() {
		return values_.data();
	}

	/**
	 *	Forward View
	 */
	inline unsigned int* row_begin() {
		return fwd_row_.data();
	}

	inline unsigned int* column_indices() {
		return fwd_col_idx_.data();
	}

	/**
	 *	Backward View
	 */
	inline unsigned int* col_begin() {
		return bwd_col_.data();
	}

	inline unsigned int* row_indices() {
		return bwd_row_idx_.data();
	}

	inline unsigned int* inverse_indices() {
		return bwd_inv_idx_.data();
	}

	int test_index(unsigned int row, unsigned int col) {
		int start = fwd_row_[row];
		int end = fwd_row_[row+1];
		if( end - start > 0 ) { // the row exist
			for(int i = start; i < end; i++) {
				if(fwd_col_idx_[i] == col)
					return i;
			}
		}

		return -1;
	}

	/**
	 * 	\brief		add all connections of one dendrite
	 * 	\details
	 * 	\param[in]	row			neuron index of afferent neuron
	 * 	\param[in]	columns		neuron indices of efferent neurons
	 * 	\param[in]	w			synaptic weights
	 */
	void push_back(int row, std::vector<int> columns, std::vector<double> w, std::vector<int> d) {
		int old_idx = fwd_row_[row+1];

		fwd_col_idx_.insert(fwd_col_idx_.begin()+old_idx, columns.begin(), columns.end());
		values_.insert(values_.begin()+old_idx, w.begin(), w.end());

		for( auto it = fwd_row_.begin()+row+1; it != fwd_row_.end(); it++ )
			*it += columns.size();
	}

	/**
	 * 	\brief		overloaded std::ostream operator<<
	 * 	\details	for the object itself
	 * 	\param[IN]	os		ostream instance
	 * 	\param[IN]	matrix	object instance
	 * 	\return		manipulated ostream instance
	 */
	friend std::ostream& operator<< (std::ostream& os, const CSRCMatrix& matrix) {
        os << "(forward)" << std::endl;
		os << "fwd_row_:" << std::endl;
		os << "[ ";
		for(unsigned int r = 0; r < matrix.num_rows_; r++) {
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

        os << "(backward)" << std::endl;
		os << "bwd_col_:" << std::endl;
		os << "[ ";
		for(unsigned int c = 0; c < matrix.num_columns_; c++) {
			os << matrix.bwd_col_[c] << " ";
		}
		os << "]" << std::endl;

		os << "bwd_row_idx: "<< std::endl;
		os << "[ ";
		for(unsigned int idx = 0; idx < matrix.values_.size(); idx++) {
			os << matrix.bwd_row_idx_[idx] << " ";
		}
		os << "]" << std::endl;
	
		os << "inv_idx:" << std::endl;
		os << "[ ";
		for(unsigned int idx = 0; idx < matrix.values_.size(); idx++) {
			os << matrix.bwd_inv_idx_[idx] << " ";
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
	friend std::ostream& operator<< (std::ostream& os, CSRCMatrix* matrix) {
		return os << *matrix;
	}
};
