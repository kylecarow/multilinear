use anyhow;
use itertools::Itertools;
use ndarray::*;

/// Generate all permutations of indices for a given *N*-dimensional array shape
///
/// # Arguments
/// * `shape` - Reference to shape of the *N*-dimensional array returned by [`ndarray::Array::shape()`]
///
/// # Returns
/// A Vec<Vec<usize>> where each inner Vec<usize> is one permutation of indices
///
/// # Examples
/// ## Example 1
/// ```rust
/// use multilinear::get_index_permutations;
/// let shape = [3, 2, 2];
/// assert_eq!(
///     get_index_permutations(&shape),
///     [
///         [0, 0, 0],
///         [0, 0, 1],
///         [0, 1, 0],
///         [0, 1, 1],
///         [1, 0, 0],
///         [1, 0, 1],
///         [1, 1, 0],
///         [1, 1, 1],
///         [2, 0, 0],
///         [2, 0, 1],
///         [2, 1, 0],
///         [2, 1, 1],
///     ]
/// );
/// ```
/// ## Example 2
/// ```rust
/// use multilinear::get_index_permutations;
/// // Empty shape
/// let shape = [];
/// assert_eq!(get_index_permutations(&shape), [[]]);
/// ```
pub fn get_index_permutations(shape: &[usize]) -> Vec<Vec<usize>> {
    if shape.is_empty() {
        return vec![vec![]];
    }
    shape
        .iter()
        .map(|&len| 0..len)
        .multi_cartesian_product()
        .collect()
}

/// Multilinear interpolation function, accepting any dimensionality *N*.
///
/// Arguments
///
/// * `point`: interpolation point - specified by *N*-length array `&[x, y, z, ...]`
///
/// * `grid`: rectilinear grid points - *N*-length array of x, y, z, ... grid coordinate vectors
///
/// * `values`: *N*-dimensional [`ndarray::ArrayD`] containing values at grid points, can be created by calling [`Array::into_dyn()`]
///
pub fn multilinear(point: &[f64], grid: &[Vec<f64>], values: &ArrayD<f64>) -> anyhow::Result<f64> {
    // Dimensionality
    let mut n = values.ndim();

    // Validate inputs
    anyhow::ensure!(
        point.len() == n,
        "Length of supplied `point` must be same as `values` dimensionality: {point:?} is not {n}-dimensional",
    );
    anyhow::ensure!(
        grid.len() == n,
        "Length of supplied `grid` must be same as `values` dimensionality: {grid:?} is not {n}-dimensional",
    );
    for i in 0..n {
        // TODO: This ensure! could be removed if subsetting got rid of length 1 dimensions in `grid` and `points` as well
        anyhow::ensure!(
            grid[i].len() > 1,
            "Supplied `grid` length must be > 1 for dimension {i}",
        );
        anyhow::ensure!(
            grid[i].len() == values.shape()[i],
            "Supplied `grid` and `values` are not compatible shapes: dimension {i}, lengths {} != {}",
            grid[i].len(),
            values.shape()[i]
        );
        anyhow::ensure!(
            grid[i].windows(2).all(|w| w[0] < w[1]),
            "Supplied `grid` coordinates must be sorted and non-repeating: dimension {i}, {:?}",
            grid[i]
        );
        anyhow::ensure!(
            grid[i][0] <= point[i] && point[i] <= *grid[i].last().unwrap(),
            "Supplied `point` must be within `grid` for dimension {i}: point[{i}] = {:?}, grid[{i}] = {:?}",
            point[i],
            grid[i],
        );
    }

    // Point can share up to N values of a grid point, which reduces the problem dimensionality
    // i.e. the point shares one of three values of a 3-D grid point, then the interpolation becomes 2-D at that slice
    // or   if the point shares two of three values of a 3-D grid point, then the interpolation becomes 1-D
    let mut point = point.to_vec();
    let mut grid = grid.to_vec();
    let mut values_view = values.view();
    for dim in (0..n).rev() {
        // Range is reversed so that removal doesn't affect indexing
        if let Some(pos) = grid[dim]
            .iter()
            .position(|&grid_point| grid_point == point[dim])
        {
            point.remove(dim);
            grid.remove(dim);
            values_view.index_axis_inplace(Axis(dim), pos);
        }
    }
    if values_view.len() == 1 {
        // Supplied point is coincident with a grid point, so just return the value
        return Ok(*values_view.first().unwrap());
    }
    // Simplified dimensionality
    n = values_view.ndim();

    // Extract the lower and upper indices for each dimension,
    // as well as the fraction of how far the supplied point is between the surrounding grid points
    let mut lower_idxs = Vec::with_capacity(n);
    let mut interp_diffs = Vec::with_capacity(n);
    for dim in 0..n {
        let lower_idx = grid[dim]
            .windows(2)
            .position(|w| w[0] < point[dim] && point[dim] < w[1])
            .unwrap();
        let interp_diff =
            (point[dim] - grid[dim][lower_idx]) / (grid[dim][lower_idx + 1] - grid[dim][lower_idx]);
        lower_idxs.push(lower_idx);
        interp_diffs.push(interp_diff);
    }
    // `interp_vals` contains all values surrounding the point of interest, starting with shape (2, 2, ...) in N dimensions
    // this gets mutated and reduces in dimension each iteration, filling with the next values to interpolate with
    // this ends up as a 0-dimensional array containing only the final interpolated value
    let mut interp_vals = values_view
        .slice_each_axis(|ax| {
            let lower = lower_idxs[ax.axis.0];
            Slice::from(lower..=lower + 1)
        })
        .to_owned();
    let mut index_permutations = get_index_permutations(&interp_vals.shape());
    // This loop interpolates in each dimension sequentially
    // each outer loop iteration the dimensionality reduces by 1
    // `interp_vals` ends up as a 0-dimensional array containing only the final interpolated value
    for dim in 0..n {
        let diff = interp_diffs[dim];
        let next_dim = n - 1 - dim;
        let next_shape = vec![2; next_dim];
        // Indeces used for saving results of this dimensions interpolation results
        // assigned to `index_permutations` at end of loop to be used for indexing in next iteration
        let next_idxs = get_index_permutations(&next_shape);
        let mut intermediate_arr = Array::default(next_shape);
        for i in 0..next_idxs.len() {
            // `next_idxs` is always half the length of `index_permutations`
            let l = index_permutations[i].as_slice();
            let u = index_permutations[next_idxs.len() + i].as_slice();
            if dim == 0 {
                anyhow::ensure!(
                    !interp_vals[l].is_nan() && !interp_vals[u].is_nan(),
                    "Surrounding value(s) cannot be NaN:\npoint = {point:?},\ngrid = {grid:?},\nvalues = {values:?}"
                );
            }
            // This calculation happens 2^(n-1) times in the first iteration of the outer loop,
            // 2^(n-2) times in the second iteration, etc.
            intermediate_arr[next_idxs[i].as_slice()] =
                interp_vals[l] * (1.0 - diff) + interp_vals[u] * diff;
        }
        index_permutations = next_idxs;
        interp_vals = intermediate_arr;
    }

    // return the only value contained within the 0-dimensional array
    Ok(*interp_vals.first().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multilinear_1d() {
        let grid = [vec![0.0, 1.0, 4.0]];
        let values = array![0.0, 2.0, 4.45].into_dyn();

        let point_a = [0.82];
        assert_eq!(multilinear(&point_a, &grid, &values).unwrap(), 1.64);

        let point_b = [2.98];
        assert_eq!(multilinear(&point_b, &grid, &values).unwrap(), 3.617);

        // returns value at x2
        let point_c = [4.0];
        assert_eq!(multilinear(&point_c, &grid, &values).unwrap(), values[2]);
    }

    // test targets found using https://www.omnicalculator.com/math/bilinear-interpolation
    #[test]
    fn test_multilinear_2d() {
        let grid = [
            vec![0.0, 1.0, 2.0], // x0, x1, x2
            vec![0.0, 1.0, 2.0], // y0, y1, y2
        ];
        let values = array![
            [0.0, 2.0, 1.9], // (x0, y0), (x0, y1), (x0, y2)
            [2.0, 4.0, 3.1], // (x1, y0), (x1, y1), (x1, y2)
            [5.0, 0.0, 1.4], // (x2, y0), (x2, y1), (x2, y2)
        ]
        .into_dyn();

        let point_a = [0.5, 0.5];
        assert_eq!(multilinear(&point_a, &grid, &values).unwrap(), 2.0);

        let point_b = [1.52, 0.36];
        assert_eq!(multilinear(&point_b, &grid, &values).unwrap(), 2.9696);

        // returns value at (x2, y1)
        let point_c = [2.0, 1.0];
        assert_eq!(
            multilinear(&point_c, &grid, &values).unwrap(),
            values[[2, 1]]
        );
    }

    #[test]
    fn test_multilinear_3d() {
        let grid = [
            vec![0.0, 1.0, 2.0], // x0, x1, x2
            vec![0.0, 1.0, 2.0], // y0, y1, y2
            vec![0.0, 1.0, 2.0], // z0, z1, z2
        ];
        let values = array![
            [
                [0.0, 1.5, 3.0], // (x0, y0, z0), (x0, y0, z1), (x0, y0, z2)
                [2.0, 0.5, 1.4], // (x0, y1, z0), (x0, y1, z1), (x0, y1, z2)
                [1.9, 5.3, 2.2], // (x0, y2, z0), (x0, y0, z1), (x0, y2, z2)
            ],
            [
                [2.0, 5.1, 1.1], // (x1, y0, z0), (x1, y0, z1), (x1, y0, z2)
                [4.0, 1.0, 0.5], // (x1, y1, z0), (x1, y1, z1), (x1, y1, z2)
                [3.1, 0.9, 1.2], // (x1, y2, z0), (x1, y2, z1), (x1, y2, z2)
            ],
            [
                [5.0, 0.2, 5.1], // (x2, y0, z0), (x2, y0, z1), (x2, y0, z2)
                [0.7, 0.1, 3.2], // (x2, y1, z0), (x2, y1, z1), (x2, y1, z2)
                [1.4, 1.1, 0.0], // (x2, y2, z0), (x2, y2, z1), (x2, y2, z2)
            ],
        ]
        .into_dyn();

        let point_a = [0.5, 0.5, 0.5];
        assert_eq!(multilinear(&point_a, &grid, &values).unwrap(), 2.0125);

        let point_b = [1.52, 0.36, 0.5];
        assert_eq!(multilinear(&point_b, &grid, &values).unwrap(), 2.46272);

        // returns value at (x2, y1, z0)
        let point_c = [2.0, 1.0, 0.0];
        assert_eq!(
            multilinear(&point_c, &grid, &values).unwrap(),
            values[[2, 1, 0]]
        );
    }

    #[test]
    fn test_multilinear_with_nans() {
        let grid = [
            vec![0.0, 1.0, 2.0, 3.0, 4.0], // x0, x1, x2, x3, x4
            vec![0.0, 1.0, 2.0, 3.0],      // y0, y1, y2, y3
        ];
        let values = array![
            [0.000000, 2.000000, 1.900000, 4.200000], // (x0, y0), (x0, y1), (x0, y2), (x0, y3)
            [2.000000, 4.000000, 3.100000, 6.100000], // (x1, y0), (x1, y1), (x1, y2), (x1, y3)
            [f64::NAN, 0.000000, 1.400000, 1.100000], // (x2, y0), (x2, y1), (x2, y2), (x2, y3)
            [f64::NAN, 0.000000, f64::NAN, f64::NAN], // (x3, y0), (x3, y1), (x3, y2), (x3, y3)
            [f64::NAN, f64::NAN, f64::NAN, f64::NAN], // (x4, y0), (x4, y1), (x4, y2), (x4, y3)
        ]
        .into_dyn();

        let point_a = [0.51, 0.36];
        assert_eq!(multilinear(&point_a, &grid, &values).unwrap(), 1.74);

        let point_b = [1.5, 2.5];
        assert_eq!(multilinear(&point_b, &grid, &values).unwrap(), 2.925);

        let point_c = [1.5, 0.5];
        assert!(multilinear(&point_c, &grid, &values).is_err());

        let point_d = [3.5, 2.5];
        assert!(multilinear(&point_d, &grid, &values).is_err());
    }
}
