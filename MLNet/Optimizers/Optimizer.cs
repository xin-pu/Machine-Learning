using Numpy;

namespace MLNet.Optimizers
{
    public abstract class Optimizer
    {
        protected Optimizer()
        {
            Name = GetType().Name;
        }

        public string Name { protected set; get; }
        public double LearningRate { set; get; } = 1E-4;


        public NDarray Call(NDarray weight, NDarray grad)
        {
            return call(weight, grad);
        }

        internal abstract NDarray call(NDarray weight, NDarray grad);
    }
}