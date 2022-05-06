using System.Text;
using AutoDiff;
using Numpy;

namespace MLNet.Losses
{
    /// <summary>
    ///     This is abstract loss base
    /// </summary>
    public abstract class Loss
    {
        protected Loss(Variable[] variables, NDarray x, NDarray y)
        {
            Name = GetType().Name;
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

        public Tuple<NDarray, double> Call(NDarray weights)
        {
            var points = weights.GetData<double>();
            var loss = CostFunc.Evaluate(Variables, points);
            var grad = CostFunc.Differentiate(Variables, points);
            return new Tuple<NDarray, double>(grad, loss);
        }

        internal abstract Term createLoss(Variable[] w, NDarray x, NDarray y);


        public override string ToString()
        {
            var str = new StringBuilder();
            str.AppendLine($"---{Name}---");
            return str.ToString();
        }
    }
}