from pathlib import Path
import json
from typing import Dict, Any, Set, Optional
from datetime import datetime
import logging

class StateManager:
    """
    Менеджер состояния обработки тем.
    Сохраняет прогресс (обработанные и неудачные темы), статистику, поддерживает recoverability.
    """

    def __init__(self, state_file: Path):
        """
        :param state_file: Путь к JSON-файлу состояния.
        """
        self.state_file = state_file
        self.logger = logging.getLogger("state_manager")
        self.state: Dict[str, Any] = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Загрузка состояния из файла или создание по умолчанию."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Преобразуем processed_topics к set для быстрого поиска
                    if isinstance(data.get("processed_topics"), list):
                        data["processed_topics"] = set(data["processed_topics"])
                    elif isinstance(data.get("processed_topics"), set):
                        pass
                    else:
                        data["processed_topics"] = set()
                    return data
            except Exception as e:
                self.logger.error(f"Failed to load state: {e}")
                return self._create_default_state()
        return self._create_default_state()

    def _create_default_state(self) -> Dict[str, Any]:
        """Создание состояния по умолчанию."""
        return {
            "last_update": datetime.utcnow().isoformat(),
            "processed_topics": set(),
            "failed_topics": {},
            "statistics": {
                "total_processed": 0,
                "successful": 0,
                "failed": 0
            }
        }

    def save_state(self) -> None:
        """Сохранение состояния (atomic)."""
        try:
            state_copy = self.state.copy()
            # Сериализуем set как list для JSON
            state_copy["processed_topics"] = list(self.state["processed_topics"])
            state_copy["last_update"] = datetime.utcnow().isoformat()
            tmp_file = self.state_file.with_suffix('.tmp')
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(state_copy, f, indent=4, ensure_ascii=False)
            tmp_file.replace(self.state_file)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            raise

    def add_processed_topic(self, topic: str) -> None:
        """Добавить обработанную тему."""
        self.state["processed_topics"].add(topic)
        self.state["statistics"]["total_processed"] += 1
        self.state["statistics"]["successful"] += 1
        self.save_state()

    def add_failed_topic(self, topic: str, error: str) -> None:
        """Добавить тему с ошибкой."""
        self.state["failed_topics"][topic] = {
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
            "attempts": self.state["failed_topics"].get(topic, {}).get("attempts", 0) + 1
        }
        self.state["statistics"]["failed"] += 1
        self.save_state()

    def get_processed_topics(self) -> Set[str]:
        """Получить множество обработанных тем."""
        return self.state["processed_topics"]

    def get_failed_topics(self) -> Dict[str, Dict[str, Any]]:
        """Получить dict тем с ошибками."""
        return self.state["failed_topics"]

    def get_statistics(self) -> Dict[str, int]:
        """Получить статистику обработки."""
        return self.state["statistics"]

    def clear_failed_topics(self) -> None:
        """Очистить список тем с ошибками."""
        self.state["failed_topics"].clear()
        self.save_state()
