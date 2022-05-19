using System.Text;
using AutoDiff;
using Numpy;

namespace MLNet.Losses
{
    public abstract class MultiClassLoss
    {
        /// <summary>
        ///     多分类损失
        /// </summary>
        /// <param name="classes"></param>
        protected MultiClassLoss(int classes)
        {
            Name = GetType().Name;
            Classes = classes;
        }

        public string Name { set; get; }

        public int Classes { get; }
        public Dictionary<int, Variable[]> Variables { protected set; get; } = null!;

        public Term CostFunc { protected set; get; } = null!;

        public void GiveVariables(Dictionary<int, Variable[]> variables)
        {
            Variables = variables;
        }

        public Tuple<NDarray, double> Call(NDarray weights, NDarray x, NDarray y)
        {
            CostFunc = createLoss(Variables, x, y);
            var points = weights.GetData<double>();

            var variablesList = Variables.SelectMany(a => a.Value).ToArray();
            var loss = CostFunc.Evaluate(variablesList, points);
            var grad = CostFunc.Differentiate(variablesList, points);
            var Grad = np.reshape(np.array(grad), weights.shape);
            return new Tuple<NDarray, double>(Grad, loss);
        }

        internal abstract Term createLoss(Dictionary<int, Variable[]> w, NDarray x, NDarray y);

        public override string ToString()
        {
            var str = new StringBuilder();
            str.AppendLine($"---{Name}---");
            return str.ToString();
        }
    }
}