"""Every human-readable string, in three languages, as parallel text.

**Parallel** is a strong word and it is meant literally: the English, Turkish and Russian versions
of a passage state the *same facts* — the same torque, the same part number, the same serial range
— so an English question and its Turkish twin have the same ground truth, each in its own
language's document. That is what makes the multilingual number in the evaluation a measurement of
the system rather than of three unrelated corpora.

The text is written in native script (Turkish diacritics, Cyrillic) rather than transliterated. A
tokeniser behaves differently on `Şanzıman` and `Sanziman`, and an evaluation run against ASCII
would be measuring a language nobody writes.

The prose is deliberately written the way a service manual is written — an instruction, a
consequence, a second figure that is not the answer. Passages that state one fact in one sentence
make retrieval look better than it is, because there is nothing in the chunk to be distracted by.

Templates take a fixed keyword set: `variant`, `value`, `part`, `n`. Not every template uses every
one, but they are all passed, so adding a placeholder to one language and not the others fails at
format time instead of producing a corpus where the Russian passage is missing a figure.
"""

# The Turkish and Russian text below trips ruff's ambiguous-character rules on almost every
# line: `ı`, `İ`, `Н`, `б`, `а`, `р` and their neighbours are exactly the characters those
# rules warn about, and here they are the point rather than a typo. Suppressed for the file,
# because a per-string noqa on a parallel corpus is noise that hides a real one.
# ruff: noqa: RUF001, RUF002, RUF003


from __future__ import annotations

from typing import Final

from parts_answer_gate.corpus.lang import Trilingual

__all__ = [
    "ABSENT_SPECIFICATIONS",
    "APPLICABILITY",
    "BULLETIN_BODY",
    "BULLETIN_TITLE",
    "DECOY_FAMILIES",
    "HEADER_TEMPLATE",
    "MANUAL_TITLE",
    "SERIAL_ALL",
    "SERIAL_RANGE",
    "SYNTHETIC_NOTICE",
    "TOPIC_BODY",
    "TOPIC_HEADING",
    "TOPIC_QUESTION",
    "UNANSWERABLE_TEMPLATES",
]

# --------------------------------------------------------------------------- document furniture

MANUAL_TITLE: Final = Trilingual(
    en="{product} — Service Manual",
    tr="{product} — Servis El Kitabı",
    ru="{product} — Руководство по обслуживанию",
)

BULLETIN_TITLE: Final = Trilingual(
    en="{product} — Field Service Bulletin",
    tr="{product} — Saha Servis Bülteni",
    ru="{product} — Сервисный бюллетень",
)

#: ADR-001: every generated file says it is synthetic in its own body. The documents say it in
#: their own text as well, so a passage quoted out of the corpus into a slide or an issue carries
#: the disclaimer with it rather than leaving it behind in a manifest.
SYNTHETIC_NOTICE: Final = Trilingual(
    en=(
        "Synthetic documentation generated for evaluation. This is not a manufacturer publication "
        "and must not be used on real equipment."
    ),
    tr=(
        "Değerlendirme amacıyla üretilmiş sentetik dokümantasyon. Bu bir üretici yayını değildir "
        "ve gerçek ekipmanda kullanılmamalıdır."
    ),
    ru=(
        "Синтетическая документация, созданная для оценки. Не является публикацией производителя "
        "и не должна применяться на реальном оборудовании."
    ),
)

#: The block every document opens with. Its rendered length is what the first chunk's offset is
#: past, so it is part of the offset contract rather than decoration.
HEADER_TEMPLATE: Final = Trilingual(
    en=(
        "{title}\nRevision {revision} · Document {document_id}\n"
        "In force from {valid_from}\n{notice}"
    ),
    tr=(
        "{title}\nRevizyon {revision} · Doküman {document_id}\n"
        "{valid_from} tarihinden itibaren geçerlidir\n{notice}"
    ),
    ru=(
        "{title}\nРедакция {revision} · Документ {document_id}\n"
        "Действует с {valid_from}\n{notice}"
    ),
)

SERIAL_ALL: Final = Trilingual(
    en="all serial numbers",
    tr="tüm seri numaraları",
    ru="все серийные номера",
)

SERIAL_RANGE: Final = Trilingual(
    en="serial numbers {first} to {last}",
    tr="{first}–{last} seri numaraları",
    ru="серийные номера {first}–{last}",
)

#: Effectivity stated in the prose as well as in the metadata. A reader who is handed the passage
#: alone must be able to see which machines it covers; a chunk whose applicability lives only in a
#: database column reads as universally true the moment it is quoted anywhere.
APPLICABILITY: Final = Trilingual(
    en="Applies to {variant}, {serials}.",
    tr="{variant} için geçerlidir, {serials}.",
    ru="Относится к {variant}, {serials}.",
)

# ------------------------------------------------------------------------------ topic headings

TOPIC_HEADING: Final[dict[str, Trilingual]] = {
    "drive_coupling_torque": Trilingual(
        en="Drive coupling bolt torque",
        tr="Tahrik kaplini cıvata torku",
        ru="Момент затяжки болтов приводной муфты",
    ),
    "filter_element": Trilingual(
        en="Return-line filter element",
        tr="Dönüş hattı filtre elemanı",
        ru="Фильтрующий элемент линии возврата",
    ),
    "reservoir_capacity": Trilingual(
        en="Hydraulic reservoir capacity",
        tr="Hidrolik depo kapasitesi",
        ru="Ёмкость гидравлического бака",
    ),
    "seal_kit": Trilingual(
        en="Cylinder seal kit",
        tr="Silindir salmastra takımı",
        ru="Комплект уплотнений цилиндра",
    ),
    "calibration_interval": Trilingual(
        en="Pressure sensor calibration interval",
        tr="Basınç sensörü kalibrasyon aralığı",
        ru="Интервал калибровки датчика давления",
    ),
    "control_fuse": Trilingual(
        en="Control circuit fuse rating",
        tr="Kumanda devresi sigorta değeri",
        ru="Номинал предохранителя цепи управления",
    ),
    "pump_shaft_bearing": Trilingual(
        en="Pump shaft bearing",
        tr="Pompa mili rulmanı",
        ru="Подшипник вала насоса",
    ),
    "relief_valve_setting": Trilingual(
        en="Relief valve setting",
        tr="Emniyet valfi ayarı",
        ru="Настройка предохранительного клапана",
    ),
    "gearbox_oil_grade": Trilingual(
        en="Gearbox oil grade",
        tr="Şanzıman yağı sınıfı",
        ru="Класс масла редуктора",
    ),
    "inspection_serial_note": Trilingual(
        en="Serial-limited mounting flange inspection",
        tr="Seri numarası sınırlı bağlantı flanşı muayenesi",
        ru="Проверка монтажного фланца с ограничением по серийным номерам",
    ),
}

# --------------------------------------------------------------------------------- topic bodies

TOPIC_BODY: Final[dict[str, Trilingual]] = {
    "drive_coupling_torque": Trilingual(
        en=(
            "Tighten the drive coupling bolts on the {variant} to {value} in a diagonal sequence. "
            "Use fastener set {part} only; a fastener of a lower grade will not hold this figure. "
            "Re-check the torque after the first {n} operating hours."
        ),
        tr=(
            "{variant} üzerindeki tahrik kaplini cıvatalarını çapraz sırayla {value} değerine "
            "sıkın. Yalnızca {part} bağlantı elemanı takımını kullanın; daha düşük sınıftaki bir "
            "cıvata bu değeri taşımaz. İlk {n} çalışma saatinden sonra torku yeniden kontrol edin."
        ),
        ru=(
            "Затяните болты приводной муфты {variant} моментом {value} крест-накрест. Применяйте "
            "только комплект крепежа {part}; крепёж более низкого класса прочности это значение не "
            "удержит. Повторно проверьте момент через {n} часов работы."
        ),
    ),
    "filter_element": Trilingual(
        en=(
            "The return-line filter element for the {variant} is {value}. Replace it together with "
            "sealing ring {part} at every service interval; the ring is not reusable. A clogged "
            "element raises case pressure and will fail the pump within {n} hours."
        ),
        tr=(
            "{variant} için dönüş hattı filtre elemanı: {value}. Her servis aralığında {part} "
            "sızdırmazlık halkasıyla birlikte değiştirin; halka tekrar kullanılamaz. Tıkanmış "
            "eleman karter basıncını yükseltir ve pompayı {n} saat içinde arızalandırır."
        ),
        ru=(
            "Фильтрующий элемент линии возврата для {variant} — {value}. Заменяйте его вместе с "
            "уплотнительным кольцом {part} при каждом обслуживании; кольцо повторно не "
            "применяется. Засорённый элемент повышает давление в корпусе и выводит насос из строя "
            "за {n} часов."
        ),
    ),
    "reservoir_capacity": Trilingual(
        en=(
            "The hydraulic reservoir of the {variant} holds {value} to the upper sight-glass mark. "
            "Fill through filler cap {part}; filling through the return port bypasses the screen. "
            "Do not exceed the mark — expansion at {n} °C vents oil through the breather."
        ),
        tr=(
            "{variant} hidrolik deposu üst gözetleme camı işaretine kadar {value} alır. Dolumu "
            "{part} dolum kapağından yapın; dönüş portundan dolum süzgeci devre dışı bırakır. "
            "İşareti aşmayın — {n} °C sıcaklıkta genleşme yağı hava tahliyesinden dışarı atar."
        ),
        ru=(
            "Гидравлический бак {variant} вмещает {value} до верхней метки смотрового стекла. "
            "Заправляйте через заливную пробку {part}; заправка через сливной порт минует сетчатый "
            "фильтр. Не превышайте метку — расширение при {n} °C выбрасывает масло через сапун."
        ),
    ),
    "seal_kit": Trilingual(
        en=(
            "Cylinder seal kit for the {variant}: {value}. The kit contains the rod seal, the "
            "wiper and back-up ring {part}. Kits for the other variants share the {n} mm rod "
            "diameter but not the gland depth and must not be substituted."
        ),
        tr=(
            "{variant} için silindir salmastra takımı: {value}. Takım mil keçesini, sıyırıcıyı ve "
            "{part} destek halkasını içerir. Diğer varyantların takımları aynı {n} mm mil çapına "
            "sahiptir ancak salmastra yuvası derinliği farklıdır ve yerine kullanılamaz."
        ),
        ru=(
            "Комплект уплотнений цилиндра для {variant}: {value}. В комплект входят уплотнение "
            "штока, грязесъёмник и опорное кольцо {part}. Комплекты других исполнений имеют тот же "
            "диаметр штока {n} мм, но иную глубину гнезда и не взаимозаменяемы."
        ),
    ),
    "calibration_interval": Trilingual(
        en=(
            "Calibrate the pressure sensor of the {variant} every {value}, and after any event "
            "that exceeded the relief setting. Record the result on form {part} and keep it for "
            "{n} years. An out-of-calibration sensor reads low and hides an over-pressure event."
        ),
        tr=(
            "{variant} basınç sensörünü her {value} bir ve emniyet ayarını aşan her olaydan sonra "
            "kalibre edin. Sonucu {part} formuna kaydedin ve {n} yıl saklayın. Kalibrasyonu "
            "bozulmuş sensör düşük okur ve aşırı basınç olayını gizler."
        ),
        ru=(
            "Калибруйте датчик давления {variant} каждые {value}, а также после каждого "
            "превышения настройки предохранительного клапана. Результат заносите в форму {part} и "
            "храните {n} года. Датчик с нарушенной калибровкой занижает показания и скрывает факт "
            "превышения давления."
        ),
    ),
    "control_fuse": Trilingual(
        en=(
            "The control circuit of the {variant} is protected by a {value} fuse. Use holder "
            "{part}; a larger fuse does not protect the {n} V solenoid loom and has caused harness "
            "fires on this circuit."
        ),
        tr=(
            "{variant} kumanda devresi {value} sigorta ile korunur. {part} sigorta yuvasını "
            "kullanın; daha büyük bir sigorta {n} V selenoid tesisatını korumaz ve bu devrede "
            "kablo yangınlarına yol açmıştır."
        ),
        ru=(
            "Цепь управления {variant} защищена предохранителем {value}. Используйте держатель "
            "{part}; предохранитель большего номинала не защищает жгут соленоидов {n} В и уже "
            "приводил к возгоранию проводки в этой цепи."
        ),
    ),
    "pump_shaft_bearing": Trilingual(
        en=(
            "Pump shaft bearing for the {variant}: {value}. Press it on against the inner race "
            "only, using tool {part}. Heating above {n} °C destroys the cage and the bearing fails "
            "within one shift."
        ),
        tr=(
            "{variant} için pompa mili rulmanı: {value}. Yalnızca iç bileziğe bastırarak {part} "
            "takımıyla monte edin. {n} °C üzerinde ısıtmak kafesi bozar ve rulman bir vardiya "
            "içinde arızalanır."
        ),
        ru=(
            "Подшипник вала насоса для {variant}: {value}. Запрессовывайте только по внутреннему "
            "кольцу с помощью приспособления {part}. Нагрев выше {n} °C разрушает сепаратор, и "
            "подшипник выходит из строя в течение одной смены."
        ),
    ),
    "relief_valve_setting": Trilingual(
        en=(
            "Set the relief valve of the {variant} to {value} at {n} l/min flow, with the oil at "
            "working temperature. Seal the adjuster with wire and lead seal {part} after setting. "
            "A setting taken cold reads high and leaves the circuit unprotected."
        ),
        tr=(
            "{variant} emniyet valfini, yağ çalışma sıcaklığındayken {n} l/dak debide {value} "
            "değerine ayarlayın. Ayardan sonra ayar vidasını tel ve {part} kurşun mühürle "
            "mühürleyin. Soğukken alınan ayar yüksek okunur ve devreyi korumasız bırakır."
        ),
        ru=(
            "Настройте предохранительный клапан {variant} на {value} при расходе {n} л/мин и "
            "рабочей температуре масла. После настройки опломбируйте регулятор проволокой и "
            "пломбой {part}. Настройка на холодном масле завышает показание и оставляет контур "
            "незащищённым."
        ),
    ),
    "gearbox_oil_grade": Trilingual(
        en=(
            "The gearbox of the {variant} takes {value} oil. Drain plug magnet {part} is inspected "
            "at the same interval. Mixing grades raises the pour point and the unit will not start "
            "below minus {n} °C."
        ),
        tr=(
            "{variant} şanzımanı {value} sınıfı yağ kullanır. {part} tahliye tapası mıknatısı aynı "
            "aralıkta kontrol edilir. Sınıfların karıştırılması akma noktasını yükseltir ve ünite "
            "eksi {n} °C altında çalışmaz."
        ),
        ru=(
            "Редуктор {variant} заправляется маслом {value}. Магнит сливной пробки {part} "
            "проверяется с той же периодичностью. Смешивание классов повышает температуру "
            "застывания, и агрегат не запускается ниже минус {n} °C."
        ),
    ),
    "inspection_serial_note": Trilingual(
        en=(
            "Inspect the mounting flange of the {variant} with inspection kit {value} before "
            "return to service. Replace bolt set {part} whenever the flange is disturbed, and "
            "torque it in {n} stages. Units outside the serial range stated below are unaffected."
        ),
        tr=(
            "{variant} bağlantı flanşını servise almadan önce {value} muayene kiti ile kontrol "
            "edin. Flanş söküldüğünde {part} cıvata takımını daima değiştirin ve {n} kademede "
            "sıkın. Aşağıda belirtilen seri aralığı dışındaki üniteler etkilenmemiştir."
        ),
        ru=(
            "Перед вводом в эксплуатацию проверьте монтажный фланец {variant} с помощью комплекта "
            "{value}. При каждой разборке фланца заменяйте комплект болтов {part} и затягивайте "
            "его в {n} приёма. Агрегаты вне указанного ниже диапазона серийных номеров не "
            "затронуты."
        ),
    ),
}

BULLETIN_BODY: Final = Trilingual(
    en=(
        "Field measurement record for the {variant}: {heading} is recorded as {value}. The figure "
        "was taken during a fleet inspection covering {n} units and has not been reconciled with "
        "the service manual revision currently in force."
    ),
    tr=(
        "{variant} için saha ölçüm kaydı: {heading} değeri {value} olarak kaydedilmiştir. Değer, "
        "{n} üniteyi kapsayan bir filo muayenesi sırasında alınmıştır ve hâlen yürürlükte olan "
        "servis el kitabı revizyonu ile uzlaştırılmamıştır."
    ),
    ru=(
        "Запись полевого измерения для {variant}: параметр «{heading}» зафиксирован как {value}. "
        "Значение получено при осмотре парка из {n} машин и не согласовано с действующей редакцией "
        "руководства по обслуживанию."
    ),
)

# ------------------------------------------------------------------------- answerable questions

TOPIC_QUESTION: Final[dict[str, Trilingual]] = {
    "drive_coupling_torque": Trilingual(
        en="What is the drive coupling bolt torque for the {variant}?",
        tr="{variant} için tahrik kaplini cıvata torku nedir?",
        ru="Какой момент затяжки болтов приводной муфты указан для {variant}?",
    ),
    "filter_element": Trilingual(
        en="Which return-line filter element fits the {variant}?",
        tr="{variant} için hangi dönüş hattı filtre elemanı kullanılır?",
        ru="Какой фильтрующий элемент линии возврата подходит для {variant}?",
    ),
    "reservoir_capacity": Trilingual(
        en="How much oil does the hydraulic reservoir of the {variant} hold?",
        tr="{variant} hidrolik deposu ne kadar yağ alır?",
        ru="Сколько масла вмещает гидравлический бак {variant}?",
    ),
    "seal_kit": Trilingual(
        en="Which cylinder seal kit is specified for the {variant}?",
        tr="{variant} için hangi silindir salmastra takımı belirtilmiştir?",
        ru="Какой комплект уплотнений цилиндра предписан для {variant}?",
    ),
    "calibration_interval": Trilingual(
        en="How often must the pressure sensor of the {variant} be calibrated?",
        tr="{variant} basınç sensörü ne sıklıkla kalibre edilmelidir?",
        ru="Как часто необходимо калибровать датчик давления {variant}?",
    ),
    "control_fuse": Trilingual(
        en="What fuse rating protects the control circuit of the {variant}?",
        tr="{variant} kumanda devresini hangi değerde sigorta korur?",
        ru="Какой номинал предохранителя защищает цепь управления {variant}?",
    ),
    "pump_shaft_bearing": Trilingual(
        en="Which pump shaft bearing is fitted to the {variant}?",
        tr="{variant} üzerinde hangi pompa mili rulmanı kullanılır?",
        ru="Какой подшипник вала насоса устанавливается на {variant}?",
    ),
    "relief_valve_setting": Trilingual(
        en="What is the relief valve setting for the {variant}?",
        tr="{variant} için emniyet valfi ayarı nedir?",
        ru="Какова настройка предохранительного клапана для {variant}?",
    ),
    "gearbox_oil_grade": Trilingual(
        en="Which gearbox oil grade is specified for the {variant}?",
        tr="{variant} şanzımanı için hangi yağ sınıfı belirtilmiştir?",
        ru="Какой класс масла редуктора предписан для {variant}?",
    ),
    "inspection_serial_note": Trilingual(
        en="Which inspection kit is required for the mounting flange of the {variant}?",
        tr="{variant} bağlantı flanşı için hangi muayene kiti gereklidir?",
        ru="Какой комплект требуется для осмотра монтажного фланца {variant}?",
    ),
}

# ----------------------------------------------------------------------- unanswerable questions

#: Specifications no document in this corpus states, for any variant. The `absent_specification`
#: negative: a real, ordinary question about a real product, whose answer simply is not published.
ABSENT_SPECIFICATIONS: Final = (
    Trilingual(
        en="coolant expansion tank pre-charge pressure",
        tr="soğutma suyu genleşme tankı ön basıncı",
        ru="давление предварительной зарядки расширительного бака",
    ),
    Trilingual(
        en="noise emission limit at the operator position",
        tr="operatör konumundaki gürültü emisyon sınırı",
        ru="предельный уровень шума на месте оператора",
    ),
    Trilingual(
        en="lifting eye proof-load certificate number",
        tr="kaldırma kulağı yük testi sertifika numarası",
        ru="номер сертификата испытания рым-болта",
    ),
    Trilingual(
        en="protective coating dry film thickness",
        tr="koruyucu kaplama kuru film kalınlığı",
        ru="толщина сухой плёнки защитного покрытия",
    ),
    Trilingual(
        en="transport shock indicator trip threshold",
        tr="nakliye darbe göstergesi tetikleme eşiği",
        ru="порог срабатывания индикатора удара при транспортировке",
    ),
    Trilingual(
        en="paint batch number of the housing",
        tr="gövde boya parti numarası",
        ru="номер партии окраски корпуса",
    ),
)

#: Products that do not exist in this corpus at all. The `different_product_family` negative.
#: Their identifiers are deliberately shaped like the real ones, because a decoy that looks like a
#: typo is a different test from a decoy that looks like a neighbouring product line.
DECOY_FAMILIES: Final = (
    Trilingual(
        en="ZM-800 booster pump",
        tr="ZM-800 basınç yükseltme pompası",
        ru="дожимной насос ZM-800",
    ),
    Trilingual(
        en="RB-12 belt conveyor drive",
        tr="RB-12 bantlı konveyör tahriki",
        ru="привод ленточного конвейера RB-12",
    ),
    Trilingual(
        en="GT-300 process chiller",
        tr="GT-300 proses soğutucusu",
        ru="технологический чиллер GT-300",
    ),
    Trilingual(
        en="LM-45 vacuum blower",
        tr="LM-45 vakum körüğü",
        ru="вакуумная воздуходувка LM-45",
    ),
    Trilingual(
        en="PK-9 hydraulic crane slew drive",
        tr="PK-9 hidrolik vinç döndürme tahriki",
        ru="механизм поворота гидравлического крана PK-9",
    ),
    Trilingual(
        en="SD-60 sludge dewatering unit",
        tr="SD-60 çamur susuzlaştırma ünitesi",
        ru="установка обезвоживания осадка SD-60",
    ),
)

#: One template per unanswerable kind. `contradictory_sources` and
#: `attribute_absent_for_existing_product` reuse the ordinary `TOPIC_QUESTION` phrasing on purpose:
#: a negative that is recognisable from its wording tests the phrasing, not the retrieval.
UNANSWERABLE_TEMPLATES: Final[dict[str, Trilingual]] = {
    "absent_specification": Trilingual(
        en="What is the {spec} for the {variant}?",
        tr="{variant} için {spec} nedir?",
        ru="Каково значение параметра «{spec}» для {variant}?",
    ),
    "near_match_different_identifier": Trilingual(
        en="What is the specified replacement interval for part {pn} on the {variant}?",
        tr="{variant} üzerindeki {pn} parçası için belirtilen değişim aralığı nedir?",
        ru="Каков указанный интервал замены детали {pn} для {variant}?",
    ),
    "superseded_without_replacement_asked": Trilingual(
        en=(
            "Part {pn} was listed for the {variant}. What is the currently approved {heading} "
            "stated for that part?"
        ),
        tr=(
            "{pn} parçası {variant} için listelenmişti. Bu parça için hâlen onaylı {heading} "
            "nedir?"
        ),
        ru=(
            "Деталь {pn} указывалась для {variant}. Каково действующее значение «{heading}» для "
            "этой детали?"
        ),
    ),
    "different_product_family": Trilingual(
        en="What is the {heading} for the {decoy}?",
        tr="{decoy} için {heading} nedir?",
        ru="Каково значение «{heading}» для {decoy}?",
    ),
    "malformed_part_number": Trilingual(
        en=(
            "Part {pn} is stamped on the housing of the {variant}. What specification applies to "
            "it?"
        ),
        tr="{variant} gövdesinde {pn} parça numarası basılı. Buna hangi teknik değer uygulanır?",
        ru="На корпусе {variant} выбит номер детали {pn}. Какая спецификация к ней относится?",
    ),
}
