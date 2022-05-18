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
        protected Loss()
        {
            Name = GetType().Name;
        }

        public string Name { protected set; get; }

        public Term CostFunc { protected set; get; } = null!;

        public Variable[] Variables { protected set; get; } = null!;

        public void GiveVariables(Variable[] variables)
        {
            Variables = variables;
        }

        public Tuple<NDarray, double> Call(NDarray weights, NDarray x, NDarray y)
        {
            CostFunc = createLoss(Variables, x, y);
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