from BlackBoxOptimizer import Optimizer, OptimisationTypes
from Models import SquareSumModel
import numpy as np
from BlackBoxOptimizer import SimulatedAnnealingOptimizer

def save_archive_to_txt(vectors, energies, outputs, filename_prefix="archive"):
    """Сохраняет архивные данные в текстовые файлы"""
    
    # Сохраняем векторы
    with open(f"{filename_prefix}_vectors.txt", "w") as f:
        f.write("Итерация\tПараметры (разделены пробелами)\n")
        for i, vec in enumerate(vectors):
            f.write(f"{i}\t{' '.join(map(str, vec))}\n")
    
    # Сохраняем значения функции
    with open(f"{filename_prefix}_energies.txt", "w") as f:
        f.write("Итерация\tЗначение_функции\n")
        for i, energy in enumerate(energies):
            f.write(f"{i}\t{energy}\n")
    
    # Сохраняем выходные значения модели
    with open(f"{filename_prefix}_outputs.txt", "w") as f:
        f.write("Итерация\tВыходные_значения (разделены пробелами)\n")
        for i, out in enumerate(outputs):
            f.write(f"{i}\t{' '.join(map(str, out))}\n")

def test_simulated_annealing():
    target_point = np.array([1, 84, 72, 0, 1, 1])
    model = SquareSumModel(-target_point)
    
    opt = Optimizer(
        optCls=SimulatedAnnealingOptimizer,
        seed=1424,
        to_model_vec_size=6,
        from_model_vec_size=1,
        iter_limit=1000,  # Уменьшил для более компактного вывода
        external_model=model.evaluate,
        optimisation_type=OptimisationTypes.minimize,
        initial_temp=50.0,
        min_temp=1e-5,
        cooling_rate=0.98,
        step_size=0.8,
        penalty_coef=1e6  
    )
    
    opt.setVecItemLimit(0, vec_dir="to_model", min=0, max=100)
    opt.setVecItemLimit(1, vec_dir="to_model", min=0, max=100)
    opt.setVecItemLimit(2, vec_dir="to_model", min=0, max=100)
    opt.setVecItemType(3, vec_dir="to_model", new_type="bool")
    opt.setVecItemType(4, vec_dir="to_model", new_type="bool")
    opt.setVecItemType(5, vec_dir="to_model", new_type="bool")
    
    opt.setVecItemLimit(1, vec_dir="from_model", min=0, max=100)

    initial_guess = np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0])
    opt.setPreSetCadidateVec(0, initial_guess)
    
    opt.modelOptimize()
    
    result = opt.getResult()
    output_values = model.evaluate(result)
    final_value = output_values[0]
    calls_count = opt.get_usage_count()
    
    print("\nРезультаты оптимизации:")
    print(f"Истинный оптимум: {target_point}")
    print(f"Найденное решение: {result}")
    print(f"Значение функции: {final_value:.6f}")
    print(f"Число вызовов модели: {calls_count}")
    print(f"Отклонение от оптимума: {np.linalg.norm(np.array(result) - target_point):.6f}")
    print(f"Отклонение значения функции: {(final_value - model.func(target_point)):.10f}")

    # Получаем исторические данные
    archive_vectors, archive_energies, archive_outputs = opt.getOptimizer().get_archive()
    
    # Сохраняем в текстовые файлы
    save_archive_to_txt(archive_vectors, archive_energies, archive_outputs)
    
    print("\nИсторические данные сохранены в файлы:")
    print("archive_vectors.txt - все векторы-кандидаты")
    print("archive_energies.txt - значения функции")
    print("archive_outputs.txt - выходные значения модели")


if __name__ == "__main__":
    test_simulated_annealing()