using Numpy;

namespace MLNet.Optimizers
{
    /// <summary>
    ///     variables=variables-η*f'(variables)
    ///     一维梯度下降
    /// </summary>
    public class SGD : Optimizer
    {
        public SGD(double learning_rate) : base(learning_rate)
        {
        }

        internal override NDarray call(NDarray weight, NDarray grad, int epoch)
        {
            var delta = -grad * LearningRate;
            return weight + delta;
        }
    }
}