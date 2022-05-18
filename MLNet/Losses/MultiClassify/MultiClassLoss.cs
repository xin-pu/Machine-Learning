using System.Text;
using AutoDiff;
using Numpy;

namespace MLNet.Losses
{
    public abstract class MultiClassLoss
    {
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

        internal abstract Term createLoss(Dictionary<int, Variable[]> w, NDarray x, NDarray y);

        public override string ToString()
        {
            var str = new StringBuilder();
            str.AppendLine($"---{Name}---");
            return str.ToString();
        }
    }
}