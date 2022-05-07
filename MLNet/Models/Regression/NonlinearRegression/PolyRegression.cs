using AutoDiff;
using MLNet.Transforms;
using Numpy;
using Numpy.Models;

namespace MLNet.Models.Regression
{
    /// <summary>
    ///     PolynomialFeatures
    ///     y=α + β1*x + β2*x^2 + ... + βn*x^n
    /// </summary>
    public class PolyRegression : SupervisedModel
    {
        public PolyRegression(
            int degree = 1)
        {
            Degree = degree;
            Transform = new Polynomial(Degree);
        }

        public int Degree { set; get; }


        internal override Variable[] initialVariables(NDarray x, NDarray y)
        {
            var featureCount = x.shape[1];
            var variables = Enumerable.Range(0, featureCount).Select(_ => new Variable()).ToArray();
            return variables;
        }

        internal override NDarray call(NDarray x, Shape shape)
        {
            if (Resolve == null) throw new Exception("Resolve is Empty");
            var y_pred = np.matmul(x, Resolve);
            return np.reshape(y_pred, shape);
        }
    }
}