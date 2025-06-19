import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


class MaskComponentAnalyzer:
    """
    Класс для анализа пересечений связных компонентов в двух масках
    """
    
    def __init__(self):
        self.original_mask1=None
        self.original_mask2=None
        self.mask1_components = None
        self.mask2_components = None
        self.mask1_stats = None
        self.mask2_stats = None

        self.IoU_mtx=None

        self.f1_score_line=None
        self.filter_th=None

        
    
    def find_connected_components(self, mask):
        """
        Находит связные компоненты в маске
        
        Args:
            mask: двумерная бинарная маска (numpy array)
            
        Returns:
            tuple: (количество компонентов, меток компонентов, статистика)
        """
        # Убеждаемся что маска бинарная
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Находим связные компоненты
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        # Исключаем фон (метка 0)
        num_components = num_labels - 1
        
        return num_components, labels, stats, centroids
    
    def calculate_intersections(self, detected_mask, reference_mask):
        # Находим компоненты в обеих масках
        self.original_mask1=detected_mask
        self.original_mask2=reference_mask


        num1, labels1, stats1, centroids1 = self.find_connected_components(detected_mask)
        num2, labels2, stats2, centroids2 = self.find_connected_components(reference_mask)
        
        # Сохраняем результаты
        self.detected_mask_components = (num1, labels1, stats1, centroids1)
        self.reference_mask_components = (num2, labels2, stats2, centroids2)
        
        # Создаем таблицу для результатов
        intersection_data = []
        
        # Перебираем все пары компонентов
        for i in range(1, num1 + 1):  # начинаем с 1, так как 0 - это фон
            for j in range(1, num2 + 1):
                # Создаем маски для текущих компонентов
                component1_mask = (labels1 == i)
                component2_mask = (labels2 == j)
                #print(component2_mask)
                #raise Exception()
                # Вычисляем пересечение
                intersection = np.logical_and(component1_mask, component2_mask)

                intersection_area = np.sum(intersection)
                
                union=np.logical_or(component1_mask, component2_mask)
                union_area=np.sum(union)

                # Площади отдельных компонентов
                detected_area = stats1[i, cv2.CC_STAT_AREA]
                reference_area = stats2[j, cv2.CC_STAT_AREA]
                
                # Коэффициент Жаккара (IoU)
                #union_area_test=(area1+area2) if (area1+area2)<=1 else 1
                union_area_old = detected_area + reference_area - intersection_area

                print(union_area,union_area_old)
                #print(union_area_test[10:20][10:20])
                #print(union_area==union_area_test)
                #raise Exception()

                IoU = intersection_area / union_area if union_area > 0 else 0
                #main_index=intersection_area/(union_area)
                intersection_data.append({
                    'detected': i,
                    'reference': j,
                    'detected_area': detected_area,
                    'reference_area': reference_area,
                    'intersection_area': intersection_area,
                    'union_area': union_area,
                    'IoU':IoU,
                    #'Процент_от_Маска1': (intersection_area / area1 * 100) if area1 > 0 else 0,
                    #'Процент_от_Маска2': (intersection_area / area2 * 100) if area2 > 0 else 0
                })
        
        self.intersection_table = pd.DataFrame(intersection_data)
        return self.intersection_table
    
    
    def create_meashure_matrix(self):
        if self.intersection_table is None:
            raise ValueError("Сначала выполните расчет пересечений")
        
        # Создаем сводную таблицу
        matrix = self.intersection_table.pivot(
            index='reference', 
            columns='detected', 
            values='IoU'
        )
        return matrix
        
    
    def calc_f1_score(self,filtr_th):
        orig_mtx=self.IoU_mtx.values
        #(detected, referenced)
        #(reference,detected)
        min_count=np.min(orig_mtx.shape)
        count_reference,count_detected=orig_mtx.shape

        
        TP=0
        TN=None
        #FP #Предсказано что есть detected, но его нет в reference #Круг есть в detected, но нет круга в referenced
        #FN #Предсказано что нет detected, но есть в reference #нет круга в detected, но есть в referenced
        
        
        tmp_mtx=np.array(orig_mtx)
        tmp_mtx[tmp_mtx<filtr_th]=0.0
        for i in range(min_count):
            matrix=tmp_mtx
            max_value=np.max(matrix)
            if max_value>0:
                TP+=1
                max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
                row_idx, col_idx = max_idx
            
                # Удаляем строку и столбец
                result = np.delete(matrix, row_idx, axis=0)  # удаляем строку
                result = np.delete(result, col_idx, axis=1)  # удаляем столбец
                tmp_mtx=result
            
            else:
                break
        # В итоге у нас остались круги, которые есть в detected, но нет в reference -> 
        #! bad FP,FN=tmp_mtx.shape
        FN=count_reference-TP #МБ так?
        FP=count_detected - TP

        precision=TP/(TP+FP)
        recall=TP/(TP+FN)
        try:
            f1_score= 2*(precision*recall)/(precision+recall)
        except:
            f1_score=0
        return f1_score,(TP,FP,FN)





    def calc_new_score(self,det_mask,ref_mask):
        


        filtration_step=0.1
        intersection_table = self.calculate_intersections(det_mask, ref_mask)
        matrix = self.create_meashure_matrix()
        self.IoU_mtx=matrix
        out_res=[]
        filtr_step_arr=[]
        f1_score_line=[]

        orig_vals=None # TP TN FP для th=0 (т.е. для исходной матрицы)
        for filtr_th in np.arange(0,1,filtration_step):
            TP_tmp=self.calc_f1_score(filtr_th)
            if filtr_th==0.:
                orig_vals=TP_tmp
            f1_score_line.append(TP_tmp[0])
            out_res.append(TP_tmp)
            filtr_step_arr.append(filtr_th)
        print(out_res)
        print(filtr_step_arr)

        self.f1_score_line=np.array(f1_score_line)
        self.filter_th=np.array(filtr_step_arr)
        self.orig_vals=orig_vals
        return (filtr_step_arr,out_res)






    
    def visualize_results(self, figsize=(15, 10)):
        detected_mask=self.original_mask1
        reference_mask=self.original_mask2
        if self.detected_mask_components is None or self.reference_mask_components is None:
            raise ValueError("Сначала выполните расчет пересечений")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        
        # Наложение масок
        overlay = np.zeros((*detected_mask.shape, 3))
        overlay[:, :, 0] = (reference_mask > 0).astype(float)  # Красный для reference
        overlay[:, :, 2] = (detected_mask> 0).astype(float)  # Синий для detected
        
        intersec_axes=axes[0,0]
        intersec_axes.imshow(overlay)
        intersec_axes.set_title('masks intersection')
        intersec_axes.axis('off')
        red_patch = mpatches.Patch(color='blue', label='detected mask')
        blue_patch = mpatches.Patch(color='red', label='reference mask')
        intersec_patch=mpatches.Patch(color=(1,0,1), label='intersection')
        # Добавление легенды
        intersec_axes.legend(handles=[red_patch, blue_patch,intersec_patch], loc='upper right')
        
        # Компоненты с метками
        _, labels1, stats1, _ = self.detected_mask_components
        _, labels2, stats2, _ = self.reference_mask_components
        
        # Цветные компоненты маски 1
        colored_labels1 = np.random.rand(np.max(labels1) + 1, 3)
        colored_labels1[0] = [0, 0, 0]  # фон черный
        colored_components1 = colored_labels1[labels1]
        
        det_axes=axes[0,1]
        det_axes.imshow(colored_components1)
        det_axes.set_title(f'detected mask components ({np.max(labels1)} pcs.)')
        det_axes.axis('off')
        
        # Добавляем номера компонентов
        for i in range(1, np.max(labels1) + 1):
            centroid = np.mean(np.where(labels1 == i), axis=1)
            det_axes.text(centroid[1], centroid[0], str(i), 
                          color='white', fontsize=12, ha='center', va='center',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        # Цветные компоненты маски 2
        colored_labels2 = np.random.rand(np.max(labels2) + 1, 3)
        colored_labels2[0] = [0, 0, 0]  # фон черный
        colored_components2 = colored_labels2[labels2]
        
        ref_axes=axes[0,2]
        ref_axes.imshow(colored_components2)
        ref_axes.set_title(f'reference mask components ({np.max(labels2)} pcs.)')
        ref_axes.axis('off')
        
        # Добавляем номера компонентов
        for i in range(1, np.max(labels2) + 1):
            centroid = np.mean(np.where(labels2 == i), axis=1)
            ref_axes.text(centroid[1], centroid[0], str(i), 
                          color='white', fontsize=12, ha='center', va='center',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
    
    
        matrix = self.create_meashure_matrix()
        IoU_axes=axes[1,0]
        cmap='Purples' #'YlOrRd'
        im = IoU_axes.imshow(matrix.values, cmap=cmap, aspect='equal')
        f1_score,(TP,FP,FN)=self.orig_vals
        IoU_axes.set_title(f'IoU matrix\nTP:{TP} | FP:{FP} | FN:{FN} | f1_score:{round(f1_score,2)}')
        IoU_axes.set_xlabel('detected_components')
        IoU_axes.set_ylabel('reference_components')
        IoU_axes.set_xticks(range(len(matrix.columns)))
        IoU_axes.set_yticks(range(len(matrix.index)))
        IoU_axes.set_xticklabels(matrix.columns)
        IoU_axes.set_yticklabels(matrix.index)


        # Minor ticks
        IoU_axes.set_xticks(np.arange(-.5, len(matrix.columns), 1), minor=True)
        IoU_axes.set_yticks(np.arange(-.5, len(matrix.index), 1), minor=True)

        # Gridlines based on minor ticks
        IoU_axes.grid(which='minor', color='k', linestyle='-', linewidth=1)

        # Remove minor ticks
        IoU_axes.tick_params(which='minor', bottom=False, left=False)

        
        # Добавляем значения в ячейки
        for i in range(len(matrix.index)):
            for j in range(len(matrix.columns)):
                value = matrix.iloc[i, j]
                if value > 0:
                    IoU_axes.text(j, i, f'{round(value,2)}', 
                                    ha='center', va='center', 
                                    color='white' if value > matrix.values.max()/2 else 'black')
        
        plt.colorbar(im, ax=IoU_axes, label='IoU value')

        ax_f1_line=axes[1,2]
        th_arr=self.filter_th
        f1_line=self.f1_score_line
        ax_f1_line.plot(th_arr,f1_line,lw=2,c='k')
        ax_f1_line.scatter(th_arr,f1_line,s=20,c='r')
        ax_f1_line.set_title('f1 score line')
        ax_f1_line.set_xlabel('filter_threshold')
        ax_f1_line.set_ylabel('f1_score')
        ax_f1_line.grid()


        #rest axes
        axes[1,1].axis('off')
        plt.tight_layout()
        fig.savefig('1d_plot.png')
        plt.show()







def create_test_masks():
    """
    Создает тестовые маски для демонстрации
    
    Returns:
        tuple: (mask1, mask2)
    """
    # Создаем маски размером 100x100
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask2 = np.zeros((100, 100), dtype=np.uint8)
    
    # Маска 1: несколько прямоугольников и кругов
    cv2.rectangle(mask1, (10, 10), (30, 40), 255, -1)  # прямоугольник
    cv2.rectangle(mask1, (50, 60), (80, 90), 255, -1)  # еще прямоугольник
    cv2.circle(mask1, (70, 25), 15, 255, -1)           # круг
    
    # Маска 2: перекрывающиеся фигуры
    cv2.rectangle(mask2, (15, 15), (45, 35), 255, -1)  # частично перекрывает первый прямоугольник
    cv2.circle(mask2, (65, 20), 12, 255, -1)           # частично перекрывает круг
    cv2.rectangle(mask2, (40, 70), (90, 95), 255, -1)  # частично перекрывает второй прямоугольник
    cv2.circle(mask2, (20, 80), 8, 255, -1)            # отдельный круг
    
    return mask1, mask2


def load_png_mask(path):
    from PIL import Image
    # Загружаем изображение
    img = Image.open(path)

    # Преобразуем в numpy массив
    img_array = np.array(img)

    # Если изображение цветное, преобразуем в оттенки серого
    if len(img_array.shape) == 3:
        img_gray = np.mean(img_array, axis=2)
    else:
        img_gray = img_array

    # Создаем булеву маску (True для одного цвета, False для другого)
    # Предполагаем, что один цвет светлее другого
    threshold = np.mean(img_gray)  # или используйте конкретное значение
    boolean_mask = img_gray > threshold
    
    return ~boolean_mask



def create_test_masks2():
    
    mask1=load_png_mask('mask1.png')
    mask2=load_png_mask('mask2.png')
    
    
    return mask1, mask2


def main():
    """
    Основная функция для демонстрации работы анализатора
    """
    print("Создание тестовых масок...")
    #mask1, mask2 = create_test_masks()
    detected_mask,reference_mask=create_test_masks2()
 
    analyzer=MaskComponentAnalyzer()
    analyzer.calc_new_score(detected_mask,reference_mask)
    analyzer.visualize_results()
    




if __name__ == "__main__":
    main()

def global_ss_f1_score(preds, targets, threshold=0.5, eps=1e-8):

    """
    Вычисляет среднее значение F1-score по батчу (один F1 на один пример).
    """
    ss_f1_scores = []
    filter_th_arr=None
    for i in range(preds.shape[0]):
        #Получаем 2 маски
        pred_bin = (preds[i] > threshold).float()
        target_bin = (targets[i] > 0.5).float()
        
        detected_mask=pred_bin
        reference_mask=target_bin
        analyzer=MaskComponentAnalyzer()
        analyzer.calc_new_score(detected_mask,reference_mask)

        ss_f1_line=analyzer.f1_score_line
        ss_f1_scores.append(ss_f1_line) #Добавили в список линий для каждого батча

        filter_th_arr=analyzer.filter_th

    
    ss_f1_scores=np.array(ss_f1_scores)
    ss_f1_mean=np.mean(ss_f1_scores,axis=0)
    return ss_f1_mean,filter_th_arr