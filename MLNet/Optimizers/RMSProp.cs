using Numpy;

namespace MLNet.Optimizers
{
    public class RMSProp : Optimizer
    {
        public RMSProp(double learning_rate) : base(learning_rate)
        {
        }

        internal override NDarray call(NDarray weight, NDarray grad)
        {
            throw new NotImplementedException();
        }
    }
}