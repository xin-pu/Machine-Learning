namespace MLNet.Transforms
{
    /// <summary>
    ///     核模型转换
    /// </summary>
    public abstract class Kernel : Transform
    {
        protected Kernel(KernelType kernelType = KernelType.Gauss)
        {
            KernelType = kernelType;
        }

        public KernelType KernelType { protected set; get; }
    }
}