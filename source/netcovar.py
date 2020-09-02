# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np

def nvar(dij_matrix2, p):
	"""
	dij_matrix2: Matrix of distances squared
	p: Probability distribution
	Returns: network variance
	"""
	assert type(dij_matrix2) == np.ndarray
	assert type(p) == np.ndarray
	assert dij_matrix2.shape[0]==dij_matrix2.shape[1]
	assert dij_matrix2.shape[0] == len(p)
	return 0.5*np.dot(p.T,np.dot(dij_matrix2,p))

def ncov(dij_matrix,jp, symmetric=True):
	"""
	dij_matrix: Distance matrix
	jp: Joint probability distribution matrix
	symmetric: If true, the function only returns one of the marginal's
	variance because in a symmetric joint probability distribution both
	marginals are the same.
	Returns: Network covariance and marginal's variance
	"""
	assert type(jp) == np.ndarray
	assert type(dij_matrix) == np.ndarray
	assert jp.shape[0]==jp.shape[1]
	assert dij_matrix.shape[0]==dij_matrix.shape[1]
	assert dij_matrix.shape[0] == jp.shape[0]
	p = np.sum(jp,axis=1)
	q = np.sum(jp,axis=0)
	cov = 0
	varp = 0
	varq = 0
	for i in range(jp.shape[0]):
		for j in range(jp.shape[1]):
			pi = p[i]
			pj = p[j]
			qj = q[j]
			qi = q[i]
			Pij = jp[i,j]
			dij = dij_matrix[i,j]
			cov += (pi*qj - Pij) * dij**2.0
			varp += pi*pj*dij**2.0
			varq += qi*qj*dij**2.0
	if symmetric:
		return 0.5*cov, 0.5*varp
	else:
		return 0.5*cov, 0.5*varp, 0.5*varq