using Numpy;

namespace MLNet.Regression
{
    /// <summary>
    ///     PolynomialFeatures
    ///     y=α + β1*x + β2*x^2 + ... + βn*x^n
    /// </summary>
    public class PolynomialFeatures : AbstractLinearRegression
    {
        public PolynomialFeatures(
            Constraint constraint = Constraint.None,
            int degree = 1,
            double alhpa = 0.3)
            : base("PolynomialFeatures")
        {
            Constraint = constraint;
            Degree = degree;
            Alpha = alhpa;
        }

        public int Degree { set; get; }
        public double Alpha { set; get; }

        public Constraint Constraint { set; get; }


        internal override Func<NDarray, NDarray, NDarray, NDarray> LeastSquares { get; set; }

        /// <summary>
        ///     Solve TheDa= X/Y
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public override NDarray Slove(NDarray x, NDarray y)
        {
            if (x.ndim != 2 || x.shape[1] != 1)
                throw new ArgumentException("Input X Shape should be [n,1]");
            var XPoly = CvtToPoly(x);
            switch (Constraint)
            {
                case Constraint.None:
                    return SloveNone(XPoly, y);

                case Constraint.L1:
                    return SloveL1(XPoly, y);

                case Constraint.L2:
                    return SloveL2(XPoly, y);

                default:
                    throw new ArgumentOutOfRangeException();
            }
        }


        /// <summary>
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public NDarray SloveNone(NDarray x, NDarray y)
        {
            var phiTranspose = np.transpose(x);
            var generalizedInverse = np.linalg.inv(np.matmul(phiTranspose, x));
            var res = np.matmul(np.matmul(generalizedInverse, phiTranspose), y);
            return res;
        }


        /// <summary>
        ///     L1 Solve Lasso Todo
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public NDarray SloveL1(NDarray x, NDarray y)
        {
            var phiTranspose = np.transpose(x);
            var generalizedInverse = np.linalg.inv(np.matmul(phiTranspose, x) + Alpha * np.eye(Degree + 1));
            var res = np.matmul(np.matmul(generalizedInverse, phiTranspose), y);
            return res;
        }

        /// <summary>
        ///     L2 Solve Ridge
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public NDarray SloveL2(NDarray x, NDarray y)
        {
            var phiTranspose = np.transpose(x);
            var generalizedInverse = np.linalg.inv(np.matmul(phiTranspose, x) + Alpha * np.eye(Degree + 1));
            var res = np.matmul(np.matmul(generalizedInverse, phiTranspose), y);
            return res;
        }

        internal NDarray CvtToPoly(NDarray x)
        {
            var xTranspose = np.transpose(x);
            var npX = np.ones(Degree + 1, x.shape[0]);
            Enumerable.Range(0, Degree + 1).ToList().ForEach(d =>
            {
                var row = np.ones(x.shape[0]) * d;
                npX[d] = np.power(xTranspose, row);
            });
            npX = np.transpose(npX);
            return npX;
        }
    }
}