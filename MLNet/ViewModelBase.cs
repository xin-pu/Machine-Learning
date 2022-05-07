using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace MLNet
{
    public abstract class ViewModelBase
        : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler? PropertyChanged;

        public void UpdateProperty<T>(ref T properValue, T newValue, [CallerMemberName] string propertyName = "")
        {
            if (Equals(properValue, newValue)) return;

            properValue = newValue;
            OnPropertyChanged(propertyName);
        }


        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null!)
        {
            var handler = PropertyChanged;
            handler?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}