using AutoDiff;
using MLNet.Utils;
using Numpy;
using Numpy.Models;

namespace MLNet.Models.Regression
{
    /// <summary>
    ///     多元线性回归
    ///     y=α + β1*x1 + β2*x2 + ... + βn*xn
    /// </summary>
    public class MultipleLinearRegression : SupervisedModel
    {
        /// <summary>
        ///     x => 1,x1,x2,x3,...,xN
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        internal override NDarray transform(NDarray x)
        {
            return transformer.to_linear_firstorder(x);
        }

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