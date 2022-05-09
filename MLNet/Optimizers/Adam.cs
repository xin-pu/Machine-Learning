using Numpy;

namespace MLNet.Optimizers
{
    public class Adam : Optimizer
    {
        public Adam(double learning_rate) : base(learning_rate)
        {
        }


        internal override NDarray call(NDarray weight, NDarray grad, int epoch = 0)
        {
            throw new NotImplementedException();
        }
    }
}