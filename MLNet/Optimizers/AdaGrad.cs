using Numpy;

namespace MLNet.Optimizers
{
    public class AdaGrad : Optimizer
    {
        public AdaGrad(double learning_rate) : base(learning_rate)
        {
        }

        internal override NDarray call(NDarray weight, NDarray grad)
        {
            throw new NotImplementedException();
        }
    }
}