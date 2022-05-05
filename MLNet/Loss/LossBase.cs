using AutoDiff;
using Numpy;

namespace MLNet.Loss
{
    /// <summary>
    ///     This is abstract loss base
    /// </summary>
    public abstract class LossBase
    {
        protected LossBase(string name, Variable[] variables, NDarray x, NDarray y)
        {
            Name = name;
            Variables = variables;
            CostFunc = CreateLoss(Variables, x, y);
        }

        public string Name { protected set; get; }

        public Term CostFunc { protected set; get; }

        public Variable[] Variables { protected set; get; }

        public Term CreateLoss(Variable[] variables, NDarray x, NDarray y)
        {
            return createLoss(variables, x, y);
        }

        public double Evaluate(double[] points)
        {
            return CostFunc.Evaluate(Variables, points);
        }

        public double[] Gradient(double[] points)
        {
            return CostFunc.Differentiate(Variables, points);
        }

        internal abstract Term createLoss(Variable[] w, NDarray x, NDarray y);
    }
}