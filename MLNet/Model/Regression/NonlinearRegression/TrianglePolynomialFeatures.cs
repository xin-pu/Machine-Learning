using Numpy;

namespace MLNet.Model.Regression
{
    public class TrianglePolynomialFeatures : PolynomialFeatures
    {
        public TrianglePolynomialFeatures(int degree)
            : base(degree)
        {
        }

        /// <summary>
        ///     x => 1,sin(x/2),cos(x/2),sin(2x/x),cos(2x/2)...sin(degree*x/2),cos(degree*x/2),
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        internal override NDarray transform(NDarray x)
        {
            var batch = x.shape[0];
            var features = x.shape[1];
            if (features != 1) throw new Exception("Regression for 1 dims");

            var xTranspose = np.transpose(x);
            var npX = np.ones(2 * Degree + 1, batch);
            Enumerable.Range(0, Degree).ToList().ForEach(d =>
            {
                npX[1 + 2 * d] = np.sin(d * xTranspose / 2);
                npX[2 + 2 * d] = np.cos(d * xTranspose / 2);
            });
            npX = np.transpose(npX);
            return npX;
        }
    }
}