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
        "{title}\nРедакция {revision} · Документ {document_id}\nДействует с {valid_from}\n{notice}"
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
    "flange_bolt_torque": Trilingual(
        en="Mounting flange bolt torque",
        tr="Bağlantı flanşı cıvata torku",
        ru="Момент затяжки болтов монтажного фланца",
    ),
    "mounting_foot_torque": Trilingual(
        en="Baseplate mounting foot bolt torque",
        tr="Taban plakası ayak cıvata torku",
        ru="Момент затяжки болтов опорных лап",
    ),
    "suction_strainer": Trilingual(
        en="Suction-side strainer element",
        tr="Emiş tarafı süzgeç elemanı",
        ru="Элемент всасывающего фильтра-сетки",
    ),
    "breather_filter": Trilingual(
        en="Reservoir breather filter",
        tr="Depo havalandırma filtresi",
        ru="Воздушный фильтр сапуна бака",
    ),
    "shaft_seal_kit": Trilingual(
        en="Rotating shaft seal kit",
        tr="Döner mil salmastra takımı",
        ru="Комплект уплотнений вращающегося вала",
    ),
    "valve_seal_kit": Trilingual(
        en="Control valve seal kit",
        tr="Kumanda valfi salmastra takımı",
        ru="Комплект уплотнений управляющего клапана",
    ),
    "lubrication_interval": Trilingual(
        en="Bearing regreasing interval",
        tr="Rulman gresleme aralığı",
        ru="Интервал смазывания подшипников",
    ),
    "inspection_interval": Trilingual(
        en="Scheduled inspection interval",
        tr="Planlı muayene aralığı",
        ru="Интервал планового осмотра",
    ),
    "heater_fuse": Trilingual(
        en="Anti-condensation heater fuse",
        tr="Yoğuşma önleyici ısıtıcı sigortası",
        ru="Предохранитель антиконденсатного обогревателя",
    ),
    "sensor_supply_voltage": Trilingual(
        en="Sensor supply rail voltage",
        tr="Sensör besleme hattı gerilimi",
        ru="Напряжение шины питания датчиков",
    ),
    "idler_bearing": Trilingual(
        en="Idler shaft bearing",
        tr="Avara mili rulmanı",
        ru="Подшипник промежуточного вала",
    ),
    "pilot_pressure_setting": Trilingual(
        en="Pilot circuit pressure setting",
        tr="Pilot devre basınç ayarı",
        ru="Настройка давления пилотного контура",
    ),
    "case_drain_limit": Trilingual(
        en="Case drain pressure limit",
        tr="Karter drenaj basıncı sınırı",
        ru="Предельное давление дренажа корпуса",
    ),
    "coolant_capacity": Trilingual(
        en="Cooling circuit fill volume",
        tr="Soğutma devresi dolum hacmi",
        ru="Объём заправки контура охлаждения",
    ),
    "retrofit_bracket_note": Trilingual(
        en="Serial-limited retrofit mounting bracket",
        tr="Seri numarası sınırlı sonradan takma braket",
        ru="Монтажный кронштейн дооснащения с ограничением по серийным номерам",
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
    "flange_bolt_torque": Trilingual(
        en=(
            "Tighten the pump-to-gearbox mounting flange bolts on the {variant} to {value}, "
            "working outward from the dowel in a star pattern over three passes. Fit gasket "
            "{part} dry; a lubricated or re-used gasket bleeds off preload and the joint relaxes "
            "below the specified figure within a shift. Do not torque the flange while the casing "
            "is above {n} °C — the bolts lose tension as the housing cools and the face begins to "
            "weep."
        ),
        tr=(
            "{variant} üzerindeki pompa–şanzıman bağlantı flanşı cıvatalarını, pimden dışa doğru "
            "yıldız düzeninde üç kademede {value} değerine sıkın. {part} contasını kuru olarak "
            "takın; yağlanmış veya yeniden kullanılmış bir conta ön yükü boşaltır ve birleşme "
            "yeri bir vardiya içinde belirtilen değerin altına düşer. Gövde {n} °C üzerindeyken "
            "flanşı sıkmayın — gövde soğudukça cıvatalar gerginliğini kaybeder ve yüzey "
            "sızdırmaya başlar."
        ),
        ru=(
            "На {variant} болты монтажного фланца между насосом и редуктором затягивайте моментом "
            "{value} в три прохода по звёздной схеме, от установочного штифта наружу. Прокладку "
            "{part} устанавливайте сухой; смазанная или повторно применённая прокладка снимает "
            "предварительную затяжку, и стык за одну смену опускается ниже указанного значения. "
            "Не затягивайте фланец при температуре корпуса выше {n} °C — при остывании болты "
            "теряют натяг и стык начинает потеть."
        ),
    ),
    "mounting_foot_torque": Trilingual(
        en=(
            "Tighten the baseplate mounting foot bolts of the {variant} to {value}, in diagonally "
            "opposite pairs, with the machine cold and the coupling halves disconnected. Every "
            "foot must bed fully on shim pack {part} before torque is applied; pulling a soft "
            "foot down with the bolt distorts the housing and drags the shaft out of alignment. "
            "Re-check each foot bolt after {n} hours of running, once the grout and the shim pack "
            "have bedded in."
        ),
        tr=(
            "{variant} taban plakası ayak cıvatalarını, makine soğukken ve kaplin yarımları "
            "ayrılmışken çapraz çiftler hâlinde {value} değerine sıkın. Tork uygulanmadan önce "
            "her ayak {part} şim takımına tam oturmalıdır; boşluklu bir ayağı cıvatayla aşağı "
            "çekmek gövdeyi çarpıtır ve mili eksenden kaçırır. Harç ve şim takımı oturduktan "
            "sonra, {n} saat çalışmanın ardından her ayak cıvatasını yeniden kontrol edin."
        ),
        ru=(
            "Болты опорных лап {variant} на фундаментной плите затягивайте моментом {value} "
            "крест-накрест, попарно, на холодной машине при разъединённых полумуфтах. До "
            "приложения момента каждая лапа должна полностью опираться на пакет прокладок {part}; "
            "подтягивание болтом лапы с зазором деформирует корпус и уводит вал из соосности. "
            "Через {n} часов работы, после осадки подливки и пакета прокладок, повторно проверьте "
            "каждый болт лапы."
        ),
    ),
    "suction_strainer": Trilingual(
        en=(
            "The suction-side strainer element for the {variant} is {value}. Clean it at every "
            "service interval and fit new gasket {part} on reassembly; the element itself is "
            "replaced only when the mesh is torn or distorted. A clogged strainer starves the "
            "pump inlet, and cavitation damage appears within {n} operating hours."
        ),
        tr=(
            "{variant} için emiş tarafı süzgeç elemanı: {value}. Her servis aralığında temizleyin "
            "ve montajda yeni {part} contası takın; eleman yalnızca tel eleği yırtıldığında veya "
            "deforme olduğunda değiştirilir. Tıkanmış süzgeç pompa emişini besleyemez ve {n} "
            "çalışma saati içinde kavitasyon hasarı ortaya çıkar."
        ),
        ru=(
            "Элемент всасывающего фильтра-сетки для {variant} — {value}. Очищайте его при каждом "
            "обслуживании и при сборке устанавливайте новую прокладку {part}; сам элемент "
            "заменяют только при разрыве или деформации сетки. Засорённый фильтр-сетка "
            "перекрывает подачу на вход насоса, и кавитационные повреждения появляются за {n} "
            "часов работы."
        ),
    ),
    "breather_filter": Trilingual(
        en=(
            "The reservoir breather filter for the {variant} is {value}. Replace it at every "
            "service interval together with adaptor seal {part}; the element is not washable and "
            "must not be refitted after cleaning. A clogged breather lets the reservoir draw "
            "unfiltered air past the cap seal, and the oil reaches its water limit within {n} "
            "operating hours."
        ),
        tr=(
            "{variant} için depo havalandırma filtresi: {value}. Her servis aralığında {part} "
            "adaptör contasıyla birlikte değiştirin; eleman yıkanabilir değildir ve "
            "temizlendikten sonra tekrar takılmamalıdır. Tıkanmış havalandırma filtresi deponun "
            "kapak keçesinden filtresiz hava çekmesine yol açar ve yağ {n} çalışma saati içinde "
            "su sınırına ulaşır."
        ),
        ru=(
            "Воздушный фильтр сапуна бака для {variant} — {value}. Заменяйте его при каждом "
            "обслуживании вместе с уплотнением адаптера {part}; элемент не подлежит промывке и "
            "после очистки повторно не устанавливается. Засорённый сапун заставляет бак "
            "подсасывать нефильтрованный воздух мимо уплотнения крышки, и масло достигает предела "
            "по содержанию воды за {n} часов работы."
        ),
    ),
    "shaft_seal_kit": Trilingual(
        en=(
            "Rotating shaft seal kit for the {variant}: {value}. Replace the lip seal together "
            "with wear sleeve {part} at every shaft overhaul; neither part is reusable once it "
            "has been pulled off the shaft. A seal track running hotter than {n} °C hardens the "
            "lip, and a hardened lip leaks within one shift."
        ),
        tr=(
            "{variant} için döner mil salmastra takımı: {value}. Mil keçesini her mil "
            "revizyonunda {part} aşınma kovanıyla birlikte değiştirin; milden sökülen parçaların "
            "hiçbiri tekrar kullanılamaz. {n} °C üzerinde çalışan keçe yatağı dudağı sertleştirir "
            "ve sertleşmiş dudak bir vardiya içinde sızdırır."
        ),
        ru=(
            "Комплект уплотнений вращающегося вала для {variant}: {value}. Заменяйте манжетное "
            "уплотнение вместе с защитной втулкой {part} при каждой переборке вала; снятые с вала "
            "детали повторно не применяются. Рабочая дорожка уплотнения с температурой выше {n} "
            "°C делает кромку жёсткой, а жёсткая кромка начинает пропускать масло в течение одной "
            "смены."
        ),
    ),
    "valve_seal_kit": Trilingual(
        en=(
            "Control valve seal kit for the {variant}: {value}. Replace every sealing ring in the "
            "kit whenever the spool is withdrawn and fit a new mounting gasket {part} at the same "
            "time; the gasket is not reusable. Internal leakage past a reused seal drops the "
            "pilot pressure, and the spool will not shift at flows below {n} l/min."
        ),
        tr=(
            "{variant} için kumanda valfi salmastra takımı: {value}. Sürgü söküldüğünde takımdaki "
            "tüm sızdırmazlık halkalarını değiştirin ve aynı işlemde yeni {part} montaj contasını "
            "takın; conta tekrar kullanılamaz. Yeniden kullanılan bir halkadan geçen iç kaçak "
            "pilot basıncını düşürür ve sürgü {n} l/dak altındaki debilerde konum değiştirmez."
        ),
        ru=(
            "Комплект уплотнений управляющего клапана для {variant}: {value}. При каждом "
            "извлечении золотника заменяйте все уплотнительные кольца из комплекта и одновременно "
            "устанавливайте новую монтажную прокладку {part}; прокладка повторно не применяется. "
            "Внутренняя утечка через повторно использованное кольцо снижает давление управления, "
            "и золотник не переключается при расходе ниже {n} л/мин."
        ),
    ),
    "lubrication_interval": Trilingual(
        en=(
            "Regrease the drive-end bearings of the {variant} every {value}, and after any wash- "
            "down that reached the shaft seal. Use grease cartridge {part} only, and purge the "
            "relief port until fresh grease appears at the housing. Halve the interval wherever "
            "the ambient temperature stays above {n} °C; over-greasing at that duty churns the "
            "bearing and raises its running temperature."
        ),
        tr=(
            "{variant} tahrik tarafı rulmanlarını her {value} bir ve mil keçesine ulaşan her "
            "yıkamadan sonra gresleyin. Yalnızca {part} gres kartuşunu kullanın ve gövdede taze "
            "gres görünene kadar tahliye portundan boşaltın. Ortam sıcaklığının {n} °C üzerinde "
            "kaldığı yerlerde aralığı yarıya indirin; bu yükte aşırı gresleme rulmanı döver ve "
            "çalışma sıcaklığını yükseltir."
        ),
        ru=(
            "Смазывайте подшипники приводной стороны {variant} каждые {value}, а также после "
            "каждой мойки, затронувшей уплотнение вала. Применяйте только картридж со смазкой "
            "{part} и продавливайте смазку через дренажный порт до появления свежей смазки в "
            "корпусе. При температуре окружающего воздуха выше {n} °C сокращайте интервал вдвое; "
            "избыток смазки при такой нагрузке взбивается и повышает рабочую температуру "
            "подшипника."
        ),
    ),
    "inspection_interval": Trilingual(
        en=(
            "Carry out the scheduled inspection of the {variant} every {value}, counted from "
            "commissioning and not from the last repair. Record the findings on checklist {part} "
            "and keep the signed copy for {n} years. An inspection deferred past the interval "
            "voids the previous record, and the unit is treated as uninspected until a full "
            "inspection has been completed."
        ),
        tr=(
            "{variant} planlı muayenesini, son onarımdan değil devreye alma tarihinden itibaren "
            "sayarak her {value} bir yapın. Bulguları {part} kontrol listesine kaydedin ve imzalı "
            "nüshayı {n} yıl saklayın. Aralığı aşacak şekilde ertelenen muayene önceki kaydı "
            "geçersiz kılar ve ünite, tam bir muayene tamamlanana kadar muayene edilmemiş "
            "sayılır."
        ),
        ru=(
            "Проводите плановый осмотр {variant} каждые {value}, считая от даты ввода в "
            "эксплуатацию, а не от последнего ремонта. Результаты заносите в контрольный лист "
            "{part} и храните подписанный экземпляр в течение {n} лет. Осмотр, отложенный сверх "
            "интервала, аннулирует предыдущую запись, и агрегат считается неосмотренным до "
            "проведения полного осмотра."
        ),
    ),
    "heater_fuse": Trilingual(
        en=(
            "The anti-condensation heater circuit of the {variant} is protected by a {value} "
            "fuse. Use holder {part}; the element draws its rated current continuously and a "
            "time-delay fuse of the same rating is not interchangeable here. The circuit stays "
            "live at {n} V with the main isolator open, so isolate the heater supply separately "
            "before working in the terminal box."
        ),
        tr=(
            "{variant} yoğuşma önleyici ısıtıcı devresi {value} sigorta ile korunur. {part} "
            "sigorta yuvasını kullanın; eleman anma akımını sürekli çeker ve aynı değerdeki "
            "gecikmeli sigorta burada yerine kullanılamaz. Ana şalter açıkken devre {n} V ile "
            "gerilim altında kalır; bu nedenle klemens kutusunda çalışmadan önce ısıtıcı "
            "beslemesini ayrıca izole edin."
        ),
        ru=(
            "Цепь антиконденсатного обогревателя {variant} защищена предохранителем {value}. "
            "Используйте держатель {part}; нагревательный элемент постоянно потребляет "
            "номинальный ток, и инерционный предохранитель того же номинала здесь неприменим. При "
            "разомкнутом главном рубильнике цепь остаётся под напряжением {n} В, поэтому перед "
            "работой в клеммной коробке отдельно отключайте питание обогревателя."
        ),
    ),
    "sensor_supply_voltage": Trilingual(
        en=(
            "The sensor supply rail of the {variant} is regulated to {value} at the module "
            "terminal strip. Measure it at the sensor connector with harness {part} fitted, since "
            "the drop across the loom is what the transducer actually sees. A rail that collapses "
            "under the full {n} mA sensor load points to the regulator, not to the sensor."
        ),
        tr=(
            "{variant} sensör besleme hattı, modül klemens bloğunda {value} olarak regüle edilir. "
            "Ölçümü {part} kablo demeti takılıyken sensör soketinde yapın; çünkü transdüserin "
            "gördüğü değer, demet üzerindeki düşümden sonraki değerdir. Tam {n} mA sensör yükü "
            "altında çöken bir hat, sensörü değil regülatörü işaret eder."
        ),
        ru=(
            "Шина питания датчиков {variant} стабилизирована на уровне {value} на клеммной "
            "колодке модуля. Измеряйте её на разъёме датчика при установленном жгуте {part}, "
            "поскольку преобразователь видит напряжение уже после падения на жгуте. Просадка шины "
            "под полной нагрузкой датчиков {n} мА указывает на регулятор, а не на датчик."
        ),
    ),
    "idler_bearing": Trilingual(
        en=(
            "Idler shaft bearing for the {variant}: {value}. Press it on against the inner race "
            "only and retain it with circlip {part}, which is renewed at every bearing change. A "
            "bearing driven on through the outer race brinells the track and runs above {n} °C "
            "within the first shift."
        ),
        tr=(
            "{variant} için avara mili rulmanı: {value}. Yalnızca iç bileziğe bastırarak monte "
            "edin ve her rulman değişiminde yenilenen {part} segmanı ile sabitleyin. Dış "
            "bilezikten zorlanarak takılan bir rulman yuvarlanma yüzeyini ezer ve ilk vardiya "
            "içinde {n} °C üzerine ısınır."
        ),
        ru=(
            "Подшипник промежуточного вала для {variant}: {value}. Запрессовывайте только по "
            "внутреннему кольцу и фиксируйте стопорным кольцом {part}, которое заменяется при "
            "каждой замене подшипника. Подшипник, запрессованный по наружному кольцу, "
            "продавливает дорожку качения и уже в первую смену нагревается выше {n} °C."
        ),
    ),
    "pilot_pressure_setting": Trilingual(
        en=(
            "Set the pilot circuit of the {variant} to {value} with the main circuit unloaded and "
            "the oil at working temperature. Take the reading at gauge port {part} rather than at "
            "the pump outlet; the outlet reading includes line loss at {n} l/min. A pilot "
            "pressure set below this figure will not shift the main spool under load."
        ),
        tr=(
            "{variant} pilot devresini, ana devre yüksüzken ve yağ çalışma sıcaklığındayken "
            "{value} değerine ayarlayın. Okumayı pompa çıkışından değil {part} manometre "
            "portundan alın; çıkış okuması {n} l/dak debideki hat kaybını içerir. Bu değerin "
            "altına ayarlanan pilot basıncı, yük altında ana sürgüyü hareket ettirmez."
        ),
        ru=(
            "Настройте пилотный контур {variant} на {value} при разгруженном главном контуре и "
            "рабочей температуре масла. Снимайте показание с порта манометра {part}, а не с "
            "выхода насоса; показание на выходе включает потери в линии при расходе {n} л/мин. "
            "Пилотное давление, настроенное ниже этого значения, не перемещает главный золотник "
            "под нагрузкой."
        ),
    ),
    "case_drain_limit": Trilingual(
        en=(
            "Case drain pressure on the {variant} must not exceed {value}, measured at the drain "
            "port with the oil at working temperature. Fit gauge adapter {part} in the drain "
            "line; a reading taken downstream of the cooler is low by the cooler's own loss at "
            "{n} l/min. Pressure above this figure lifts the shaft seal and the pump loses charge "
            "within one shift."
        ),
        tr=(
            "{variant} karter drenaj basıncı, yağ çalışma sıcaklığındayken drenaj portundan "
            "ölçüldüğünde {value} değerini aşmamalıdır. Drenaj hattına {part} manometre "
            "adaptörünü takın; soğutucunun çıkışından alınan okuma, {n} l/dak debideki soğutucu "
            "kaybı kadar düşük çıkar. Bu değerin üzerindeki basınç mil keçesini kaldırır ve pompa "
            "bir vardiya içinde besleme basıncını yitirir."
        ),
        ru=(
            "Давление дренажа корпуса {variant} не должно превышать {value} при измерении на "
            "дренажном порту и рабочей температуре масла. Установите в дренажную линию переходник "
            "манометра {part}; показание, снятое после охладителя, занижено на собственные потери "
            "охладителя при расходе {n} л/мин. Давление выше этого значения отрывает манжету "
            "вала, и насос теряет подпитку в течение одной смены."
        ),
    ),
    "coolant_capacity": Trilingual(
        en=(
            "The cooling circuit of the {variant} holds {value} between the minimum and maximum "
            "marks of the expansion tank. Fill through cap {part} with the circuit cold; filling "
            "hot traps air in the top of the cooler core. Do not exceed the upper mark — "
            "expansion at {n} °C discharges coolant through the relief cap."
        ),
        tr=(
            "{variant} soğutma devresi, genleşme tankındaki asgari ve azami işaretler arasında "
            "{value} alır. Devre soğukken {part} kapağından doldurun; sıcakken yapılan dolum "
            "soğutucu peteğinin üst kısmında hava hapseder. Üst işareti aşmayın — {n} °C "
            "sıcaklıkta genleşme, soğutma sıvısını basınç kapağından dışarı atar."
        ),
        ru=(
            "Контур охлаждения {variant} вмещает {value} между минимальной и максимальной метками "
            "расширительного бака. Заправляйте через пробку {part} на холодном контуре; заправка "
            "на горячем контуре оставляет воздух в верхней части сердцевины охладителя. Не "
            "превышайте верхнюю метку — расширение при {n} °C выбрасывает охлаждающую жидкость "
            "через предохранительную пробку."
        ),
    ),
    "retrofit_bracket_note": Trilingual(
        en=(
            "Machines of the {variant} in the serial range stated below were built with the short "
            "mounting foot and require retrofit bracket {value}. Install it together with shim "
            "pack {part}, which restores the shaft height the short foot does not provide, and "
            "torque the four fasteners in {n} stages. Units outside that block carry the tall "
            "foot from the factory and need no bracket."
        ),
        tr=(
            "Aşağıda belirtilen seri aralığındaki {variant} üniteleri kısa montaj ayağı ile "
            "üretilmiştir ve {value} sonradan takma braketini gerektirir. Braketi, kısa ayağın "
            "sağlamadığı mil yüksekliğini geri kazandıran {part} şim takımıyla birlikte monte "
            "edin ve dört bağlantı elemanını {n} kademede sıkın. Bu blok dışındaki üniteler "
            "fabrikadan uzun ayakla çıkmıştır ve brakete ihtiyaç duymaz."
        ),
        ru=(
            "Агрегаты {variant} из указанного ниже диапазона серийных номеров изготовлены с "
            "короткой опорной лапой и требуют кронштейна дооснащения {value}. Устанавливайте его "
            "вместе с комплектом регулировочных прокладок {part}, который восстанавливает "
            "недостающую при короткой лапе высоту вала, и затягивайте четыре крепёжных элемента в "
            "{n} приёма. Агрегаты вне этого блока выходят с завода с высокой лапой и в кронштейне "
            "не нуждаются."
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
    "flange_bolt_torque": Trilingual(
        en="What is the mounting flange bolt torque for the {variant}?",
        tr="{variant} için bağlantı flanşı cıvata torku nedir?",
        ru="Какой момент затяжки болтов монтажного фланца указан для {variant}?",
    ),
    "mounting_foot_torque": Trilingual(
        en="To what torque are the baseplate mounting foot bolts of the {variant} tightened?",
        tr="{variant} taban plakası ayak cıvataları hangi torkla sıkılır?",
        ru="Каким моментом затягиваются болты опорных лап {variant}?",
    ),
    "suction_strainer": Trilingual(
        en="Which suction-side strainer element is specified for the {variant}?",
        tr="{variant} için hangi emiş tarafı süzgeç elemanı belirtilmiştir?",
        ru="Какой элемент всасывающего фильтра-сетки предписан для {variant}?",
    ),
    "breather_filter": Trilingual(
        en="Which breather filter is fitted to the reservoir of the {variant}?",
        tr="{variant} deposuna hangi havalandırma filtresi takılır?",
        ru="Какой воздушный фильтр сапуна устанавливается на бак {variant}?",
    ),
    "shaft_seal_kit": Trilingual(
        en="Which rotating shaft seal kit is specified for the {variant}?",
        tr="{variant} için hangi döner mil salmastra takımı belirtilmiştir?",
        ru="Какой комплект уплотнений вращающегося вала предписан для {variant}?",
    ),
    "valve_seal_kit": Trilingual(
        en="Which control valve seal kit is specified for the {variant}?",
        tr="{variant} için hangi kumanda valfi salmastra takımı belirtilmiştir?",
        ru="Какой комплект уплотнений управляющего клапана предписан для {variant}?",
    ),
    "lubrication_interval": Trilingual(
        en="How often must the bearings of the {variant} be regreased?",
        tr="{variant} rulmanları ne sıklıkla greslenmelidir?",
        ru="Как часто необходимо смазывать подшипники {variant}?",
    ),
    "inspection_interval": Trilingual(
        en="How often must a scheduled inspection of the {variant} be carried out?",
        tr="{variant} planlı muayenesi ne sıklıkla yapılmalıdır?",
        ru="Как часто необходимо проводить плановый осмотр {variant}?",
    ),
    "heater_fuse": Trilingual(
        en="What fuse rating protects the anti-condensation heater circuit of the {variant}?",
        tr="{variant} yoğuşma önleyici ısıtıcı devresini hangi değerde sigorta korur?",
        ru="Какой номинал предохранителя защищает цепь антиконденсатного обогревателя {variant}?",
    ),
    "sensor_supply_voltage": Trilingual(
        en="What is the sensor supply rail voltage for the {variant}?",
        tr="{variant} için sensör besleme hattı gerilimi nedir?",
        ru="Какое напряжение шины питания датчиков указано для {variant}?",
    ),
    "idler_bearing": Trilingual(
        en="Which idler shaft bearing is fitted to the {variant}?",
        tr="{variant} üzerinde hangi avara mili rulmanı kullanılır?",
        ru="Какой подшипник промежуточного вала устанавливается на {variant}?",
    ),
    "pilot_pressure_setting": Trilingual(
        en="What is the pilot circuit pressure setting for the {variant}?",
        tr="{variant} için pilot devre basınç ayarı nedir?",
        ru="Какова настройка давления пилотного контура для {variant}?",
    ),
    "case_drain_limit": Trilingual(
        en="What is the maximum permissible case drain pressure for the {variant}?",
        tr="{variant} için izin verilen azami karter drenaj basıncı nedir?",
        ru="Какое максимально допустимое давление дренажа корпуса установлено для {variant}?",
    ),
    "coolant_capacity": Trilingual(
        en="How much coolant does the cooling circuit of the {variant} hold?",
        tr="{variant} soğutma devresi ne kadar soğutma sıvısı alır?",
        ru="Сколько охлаждающей жидкости вмещает контур охлаждения {variant}?",
    ),
    "retrofit_bracket_note": Trilingual(
        en="Which retrofit mounting bracket is required for the {variant}?",
        tr="{variant} için hangi sonradan takma montaj braketi gereklidir?",
        ru="Какой монтажный кронштейн дооснащения требуется для {variant}?",
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
            "{pn} parçası {variant} için listelenmişti. Bu parça için hâlen onaylı {heading} nedir?"
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
