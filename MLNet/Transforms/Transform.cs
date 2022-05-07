using Numpy;

namespace MLNet.Transforms
{
    /// <summary>
    ///     模型转换器
    /// </summary>
    public abstract class Transform
    {
        protected Transform()
        {
            Name = GetType().Name;
        }

        public string Name { protected set; get; }

        public abstract NDarray Call(NDarray input);

        public override string ToString()
        {
            return Name;
        }
    }
}