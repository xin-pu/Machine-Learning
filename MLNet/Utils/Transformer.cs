using Numpy;

namespace MLNet.Utils
{
    public static class transformer
    {
        /// <summary>
        ///     will return NDArray [1,x1,x2,x3,...,xN]
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static NDarray to_linear_firstorder(NDarray a)
        {
            var b = np.ones(a.shape[0]);
            var res = np.insert(a, 0, b, 1);
            return res;
        }

        /// <summary>
        ///     will return NDArray [1,x1,x2,x3,...,xN]
        /// </summary>
        /// <param name="a"></param>
        /// <returns></returns>
        public static NDarray to_kernel(NDarray a)
        {
            var b = np.ones(a.shape[0]);
            var res = np.insert(a, 0, b, 1);
            return res;
        }

        /// <summary>
        ///     x => 1,x,x^2,x^3,...,x^N
        /// </summary>
        /// <param name="x"></param>
        /// <returns>[1,x,x^2,x^3,...,x^N]</returns>
        public static NDarray to_poly(NDarray x, int degree)
        {
            var batch = x.shape[0];
            var features = x.shape[1];
            if (features != 1) throw new Exception("Regression for 1 dims");

            var xTranspose = np.transpose(x);
            var npX = np.ones(degree + 1, batch);
            Enumerable.Range(1, degree).ToList().ForEach(d =>
            {
                var row = np.ones(x.shape[0]) * d;
                npX[d] = np.power(xTranspose, row);
            });
            npX = np.transpose(npX);
            return npX;
        }

        /// <summary>
        ///     x => 1,sin(x/2),cos(x/2),sin(2x/x),cos(2x/2)...sin(degree*x/2),cos(degree*x/2),
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static NDarray to_trianglePoly(NDarray x, int degree)
        {
            var batch = x.shape[0];
            var features = x.shape[1];
            if (features != 1) throw new Exception("Regression for 1 dims");

            var xTranspose = np.transpose(x);
            var npX = np.ones(2 * degree + 1, batch);
            Enumerable.Range(0, degree).ToList().ForEach(d =>
            {
                npX[1 + 2 * d] = np.sin(d * xTranspose / 2);
                npX[2 + 2 * d] = np.cos(d * xTranspose / 2);
            });
            npX = np.transpose(npX);
            return npX;
        }
    }
}