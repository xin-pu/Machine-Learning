using AutoDiff;
using MLNet.Losses;
using MLNet.Utils;
using Numpy;

namespace MLNet.Models.Regression
{
    /// <summary>
    ///     多元线性回归
    ///     y=α + β1*x1 + β2*x2 + ... + βn*xn
    /// </summary>
    public class MultipleLinearRegression : Model
    {
        public MultipleLinearRegression(Constraint constraint = Constraint.None)

        {
            Constraint = constraint;
        }

        public Constraint Constraint { set; get; }


        /// <summary>
        ///     x => 1,x1,x2,x3,...,xN
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        internal override NDarray transform(NDarray x)
        {
            return transformer.to_linear_firstorder(x);
        }


        internal override NDarray call(NDarray x)
        {
            if (Resolve == null) throw new Exception("Resolve is Empty");
            return np.matmul(x, Resolve);
        }

        internal override Loss initialLoss(
            Variable[] variables,
            NDarray x,
            NDarray y)
        {
            return new LSLoss(variables, x, y)
            {
                Constraint = Constraint,
                Lamdba = 0.1
            };
        }
    }
}