using Numpy;

namespace MLNet.Optimizers
{
    /// <summary>
    ///     variables=variables-η*f'(variables)
    ///     一维梯度下降
    /// </summary>
    public class SGD : Optimizer
    {
        internal override NDarray call(NDarray weight, NDarray grad)
        {
            return weight - LearningRate * grad;
        }
    }
}